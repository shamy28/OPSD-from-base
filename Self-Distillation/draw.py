import json
import matplotlib.pyplot as plt
import os
import glob
import re

# 你的输出目录
BASE_DIR = "./output_sdft_olmo3chat"

def get_latest_checkpoint(base_dir):
    # 找到所有 checkpoint-xxx 文件夹
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # 按数字排序找到最大的那个
    # 例如 checkpoint-1900 > checkpoint-200
    def extract_step(path):
        match = re.search(r"checkpoint-(\d+)", path)
        return int(match.group(1)) if match else -1
    
    latest_ckpt = max(checkpoints, key=extract_step)
    return latest_ckpt

def plot_loss():
    # 1. 自动定位最新 checkpoint
    latest_ckpt = get_latest_checkpoint(BASE_DIR)
    if not latest_ckpt:
        print(f"❌ 在 {BASE_DIR} 下没找到任何 checkpoint 文件夹！")
        return
    
    print(f"📂 锁定最新 Checkpoint: {latest_ckpt}")
    json_path = os.path.join(latest_ckpt, "trainer_state.json")
    
    if not os.path.exists(json_path):
        print(f"❌ 文件不存在: {json_path}")
        return

    # 2. 读取数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    steps = []
    losses = []
    
    # 提取 log_history
    # 这是一个列表，包含 [{'loss': 0.5, 'step': 10}, {'loss': 0.4, 'step': 20}, ...]
    for entry in data['log_history']:
        if 'loss' in entry and 'step' in entry:
            steps.append(entry['step'])
            losses.append(entry['loss'])
            
    if not steps:
        print("⚠️日志为空，可能还没到第一个 logging_steps")
        return

    # 3. 画图
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, label='Training Loss', color='#1f77b4', linewidth=1.5)
    
    # 加上平滑曲线 (可选，如果抖动太厉害)
    if len(losses) > 20:
        def moving_average(a, n=5):
            ret = []
            for i in range(len(a)):
                start = max(0, i - n)
                end = min(len(a), i + n + 1)
                ret.append(sum(a[start:end]) / (end - start))
            return ret
        
        smooth_losses = moving_average(losses, n=5)
        plt.plot(steps, smooth_losses, label='Smoothed Loss', color='orange', linewidth=2, alpha=0.8)

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve (up to step {steps[-1]})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    output_img = "training_loss_curve.png"
    plt.savefig(output_img, dpi=300)
    print(f"✅ 成功！曲线图已保存为: {output_img}")
    print(f"📊 当前最新 Loss: {losses[-1]:.4f} (Step {steps[-1]})")

if __name__ == "__main__":
    plot_loss()
