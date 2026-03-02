from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
from collections import defaultdict
import random
def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_steps", type=int, default=200, help="Max training steps")
    return parser.parse_args()


# def load_tooluse_dataset(seed=42) -> Dataset:
#     """Load and prepare tooluse dataset with formatted prompts (Context Distillation)."""
#     train_path = 'data/tooluse_data/train_data.json'
#     test_path = 'data/tooluse_data/eval_data.json'
#     train_dataset = Dataset.from_json(train_path)
#     test_dataset = Dataset.from_json(test_path)

#     # --- 1. 准备 Shots (Context) ---
#     # 根据你的要求：每个 shot 前面都带一个 system prompt
#     shots_context = ""
#     for i in range(2):
#         shot_ex = train_dataset[i]
#         q = shot_ex.get('instruction', shot_ex['prompt'])
#         a = shot_ex['golden_response'][0] if isinstance(shot_ex['golden_response'], list) else shot_ex['golden_response']
        
#         # 构建 Shot 的 System Prompt (包含该 Shot 对应的工具文档)
#         shot_system = f"<|im_start|>system\nYou are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions><|im_end|>\n"
        
#         # 拼接：System + User + Assistant
#         shots_context += f"{shot_system}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|endoftext|>\n"

#     def format_example(example):
#         # --- 2. 准备当前样本 ---
#         question = example.get('instruction', example['prompt'])
        
#         # 当前问题的 System Prompt
#         current_system_prompt = f"<|im_start|>system\nYou are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions><|im_end|>\n"
        
#         # --- 3. 构建 Student Prompt (Zero-shot) ---
#         # Student 只有：System + Query
#         full_student_prompt = f"{current_system_prompt}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

#         # --- 4. 构建 Teacher Prompt (Few-shot) ---
#         # Teacher 有：Shot1 + Shot2 + System + Query
#         # Teacher 会根据前面的 Shots 模式，自己生成回答
#         full_teacher_prompt = f"{shots_context}{full_student_prompt}"

#         return {
#             "prompt": full_student_prompt,
#             "teacher_prompt": full_teacher_prompt,
#         }
    
#     train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
#     train_dataset = train_dataset.shuffle(seed=seed)
#     return train_dataset, None

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare dataset with formatted prompts (Same-Source Context Distillation)."""
    
    # 1. 改为读取你转换好的新数据
    data_files = "IF_data_single_turn.json"
    train_dataset = load_dataset("json", data_files=data_files, split="train")

    # --- [新增] 建立索引：为了能找到同源的数据 ---
    source_indices = defaultdict(list)
    for idx, item in enumerate(train_dataset):
        src = item.get('source_dataset', 'default')
        source_indices[src].append(idx)

    # 辅助函数：从 messages 里提取第一轮对话
    def get_qa_from_messages(messages):
        q, a = "", ""
        for m in messages:
            if m['role'] == 'user' and not q: q = m['content']
            elif m['role'] == 'assistant' and q and not a: a = m['content']; break
        return q, a

    def format_example(example, idx): # 注意这里多了 idx 参数
        # --- 1. 准备 Shots (动态同源采样) ---
        src = example.get('source_dataset', 'default')
        candidates = source_indices[src]
        
        # 随机抽 2 个，排除当前自己 (idx)
        # 如果同源的不够，就从全量里补(兜底)
        pool = candidates if len(candidates) > 2 else list(range(len(train_dataset)))
        shot_indices = []
        while len(shot_indices) < 2:
            rand_i = random.choice(pool)
            if rand_i != idx and rand_i not in shot_indices:
                shot_indices.append(rand_i)
        
        shots_context = ""
        for shot_idx in shot_indices:
            # 获取 shot 数据
            shot_ex = train_dataset[shot_idx]
            q, a = get_qa_from_messages(shot_ex['messages'])
            
            # [保持你的原格式] 构建 Shot
            shot_system = f"<|im_start|>system\nYou are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions><|im_end|>\n"
            shots_context += f"{shot_system}<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|endoftext|>\n"

        # --- 2. 准备当前样本 ---
        # [修改] 从 messages 列表获取当前问题
        question, _ = get_qa_from_messages(example['messages'])
        
        if not question: # 异常数据处理
            return {"prompt": "", "teacher_prompt": ""}

        # [保持你的原格式] 当前问题的 System Prompt
        current_system_prompt = f"<|im_start|>system\nYou are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions><|im_end|>\n"
        
        # --- 3. 构建 Student Prompt (Zero-shot) ---
        full_student_prompt = f"{current_system_prompt}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        # --- 4. 构建 Teacher Prompt (Few-shot) ---
        full_teacher_prompt = f"{shots_context}{full_student_prompt}"

        return {
            "prompt": full_student_prompt,
            "teacher_prompt": full_teacher_prompt,
        }
    
    # 使用 with_indices=True 把 idx 传进去
    train_dataset = train_dataset.map(format_example, with_indices=True)
    
    # 过滤掉空数据 (如果有的话)
    train_dataset = train_dataset.filter(lambda x: len(x['prompt']) > 0)
    
    # 移除多余列，只留模型需要的
    cols_to_remove = [c for c in train_dataset.column_names if c not in ['prompt', 'teacher_prompt']]
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None

if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset, _ = load_tooluse_dataset(args.seed)

    config = DistilConfig(
        seed=args.seed,
        use_vllm = True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1, 
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True, 
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = args.num_prompts_per_batch,
        max_prompt_length = 4096,
        max_completion_length = 4096,
        num_train_epochs = args.num_train_epochs,
        save_steps = 100,
        max_grad_norm = 1,
        report_to = "wandb",
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
        max_steps = args.max_steps,
        num_generations = 1,
    )
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

