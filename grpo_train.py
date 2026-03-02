from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward 

# 1. 加载数据集
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

# 2. 定义 Config
# 注意：max_completion_length 必须放在这里！
training_args = GRPOConfig(
    output_dir="output_sdft_olmo3chat_grpo",
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_generations=4,
    
    # 关键修正 1: 这个参数属于 GRPOConfig，负责控制生成的最大长度
    max_completion_length=8192, 
    
    # 关键修正 2: 删除了 max_prompt_length
    # 如果显存不够，请在 dataset.map 中手动截断 prompt
)

# 3. 初始化 Trainer
trainer = GRPOTrainer(
    model="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-900",
    reward_funcs=accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    # 这里不要再传任何 length 参数了
)

trainer.train()
