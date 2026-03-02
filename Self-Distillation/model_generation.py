import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
# 你的模型路径 (训练输出的 checkpoint 文件夹)
MODEL_PATH = "./output_sdft_50k/checkpoint-1000"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/OLMo-2-1124-7B"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_50k_Olmo2/checkpoint-200"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-200"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-900"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3math/checkpoint-400"
#MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-1025-7B"
# 或者你的 Base Model 路径
# MODEL_PATH = "/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-1025-7B"

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_response(model, tokenizer, query):
    # --- 1. 手写 Chat 模版 (与训练数据保持 100% 一致) ---
    # 你的训练代码里 system prompt 后面有个换行符 \n
    system_text = "You are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions>"
    
    # 构造 Prompt 字符串
    # 结构: [System] + [User] + [Assistant Header]
    prompt = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # --- 2. 编码与生成 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 打印一下实际输入的 Prompt，方便检查格式
    print("\n[Input Prompt]:")
    print(prompt)
    print("-" * 50)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,       # 最大生成长度
            do_sample=True,           # 开启采样 (False 为贪婪搜索)
            temperature=0.7,          # 温度
            top_p=0.9,                # 核采样
            
            # [关键] 设置停止符
            # 你的训练目标是 <|endoftext|>，同时也防止它生成 user 标签
            eos_token_id=[
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>")
            ],
            pad_token_id=tokenizer.pad_token_id
        )

    # --- 3. 解码与切分 ---
    # 只解码新增的部分 (Output)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return response

def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    print("模型加载完毕！输入 'exit' 退出。")

    while True:
        query = input("\nUser: ")
        if query.strip().lower() == "exit":
            break
        
        response = generate_response(model, tokenizer, query)
        
        # 清洗一下显示的特殊字符，方便阅读
        display_response = response.replace("<|endoftext|>", "[EOS]")
        print(f"\nAssistant: {display_response}")

if __name__ == "__main__":
    main()
