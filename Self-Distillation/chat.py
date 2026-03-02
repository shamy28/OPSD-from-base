import json
import os
from transformers import AutoTokenizer

# 1. 指向你训练好的 checkpoint 路径
MODEL_PATH = "./output_sdft_50k/checkpoint-600"

def inject_chatml_template():
    print(f"正在加载 Tokenizer: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ================= 核心：你的训练模版 =================
    # 逻辑完全对应你训练时的 format_example 函数：
    # 1. 开头不需要自动加 System（因为你训练数据里可能自带，或者这里加上默认的）
    # 2. User 之前加 <|im_start|>user\n
    # 3. Content 之后加 <|im_end|>\n
    # 4. Assistant 之前加 <|im_start|>assistant\n
    
    # Jinja2 模板解释：
    # - 遍历 messages
    # - 如果是 system/user/assistant，格式化为 <|im_start|>role\nContent<|im_end|>\n
    # - loop.last 检查：如果是最后一条且 add_generation_prompt=True，则生成 assistant 头
    
    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    
    # 强制设置 chat_template
    tokenizer.chat_template = chat_template
    
    # 2. 确保特殊 Token 存在 (Olmo 可能没有 im_start/im_end)
    # 如果你的 base model tokenizer 没有这些 token，必须加上，否则 eval 会报错或乱码
    special_tokens_dict = {
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<functions>", "</functions>"]
    }
    # 注意：如果训练时已经扩充了词表，这里就不需要 add_tokens，只需要确认它们在就行
    # 如果训练时没有扩充词表（只是当作普通字符训练的），那这里也不要加，否则会导致 id 错位
    
    print(f"当前 Tokenizer 词表大小: {len(tokenizer)}")
    print("注入 Chat Template...")

    # 3. 保存回 checkpoint 目录
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"✅ 成功！Chat Config 已注入到 {MODEL_PATH}")

    # ================= 验证环节 =================
    print("\n[Self-Check] 模拟生成 Prompt:")
    messages = [
        {"role": "system", "content": "You are a helpful function-calling AI assistant.\nYou do not currently have access to any functions. <functions></functions>"},
        {"role": "user", "content": "Who would win in a fight - a dinosaur or a cow named Moo Moo?"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    expected_end = "<|im_start|>assistant\n"
    if prompt.endswith(expected_end):
        print("✅ 格式验证通过！结尾正确。")
    else:
        print("❌ 格式警告：结尾看起来不对，请检查 Jinja2 模板。")

if __name__ == "__main__":
    inject_chatml_template()
