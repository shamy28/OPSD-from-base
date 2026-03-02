import os
from transformers import AutoTokenizer

# 你的 checkpoint 路径
MODEL_PATH = "./output_sdft_olmo3chat/checkpoint-1000"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-7B-Instruct"
# MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-1025-7B"
MODEL_PATH="/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3math/checkpoint-400"
def inject_hardcoded_system_prompt():
    print(f"正在加载: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ================= 核心修改 =================
    # 我们不再检查 messages[0] 是不是 system
    # 而是直接把 System Prompt 字符串写死在模板的最开头！
    # 这样 lm-eval 只要一调用，开头永远是这个 System Prompt
    
    SYSTEM_TEXT = "You are a helpful function-calling AI assistant.\\nYou do not currently have access to any functions. <functions></functions>"
    
    chat_template = (
        # 1. 强制输出 System Prompt (注意 im_start 和 im_end)
        f"{{{{ '<|im_start|>system\\n{SYSTEM_TEXT}<|im_end|>\\n' }}}}"
        
        # 2. 然后再遍历 messages (通常只有 user/assistant)
        "{% for message in messages %}"
            # 防止 lm-eval 偶尔真的传了 system (虽然概率极低)，如果传了就跳过，避免重复
            "{% if message['role'] != 'system' %}"
                "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
            "{% endif %}"
        "{% endfor %}"
        
        # 3. 生成引导符
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{% endif %}"
    )

    tokenizer.chat_template = chat_template
    
    # 确保特殊 token 存在
    # (如果你的词表里本来就有这些，这步会自动跳过；如果没有，最好加上以免报错)
    special_tokens = ["<|im_start|>", "<|im_end|>", "<functions>", "</functions>"]
    # tokenizer.add_tokens(special_tokens) # 视情况取消注释，通常SDFT不需要动词表

    tokenizer.save_pretrained(MODEL_PATH)
    print(f"✅ 已强制植入 System Prompt 到 {MODEL_PATH}")

    # ================= 验证 =================
    print("\n[模拟 lm-eval 行为] 输入只有 User:")
    # 模拟 lm-eval 的输入 (只有 user)
    test_msgs = [{"role": "user", "content": "Calculate 1+1."}]
    
    prompt = tokenizer.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
    
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    if "You are a helpful function-calling AI assistant" in prompt:
        print("✅ 验证成功：虽然输入没带 System，但输出里自动补上了！")
    else:
        print("❌ 验证失败：System Prompt 还是没出来。")

if __name__ == "__main__":
    inject_hardcoded_system_prompt()
