#!/bin/bash

# ================= 配置区域 =================
# 显卡设置
export CUDA_VISIBLE_DEVICES="0,1"

# 后端选择：强烈建议使用 vllm 以利用 H200 性能
# 如果报错 "vllm not found"，请执行 pip install vllm，或者将此处改为 "hf"
BACKEND="vllm" 

# 输出目录
LOG_DIR="log_chatmodel"
OUTPUT_PATH="mt_chatmodel"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_PATH"

# 模型列表
models=(
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_50k/checkpoint-600"
"/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-7B-Instruct"
"/data/users/zhenyugao/sjd/self_distlitation/Olmo-3-1025-7B"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-900"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-1000"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-1900"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3chat/checkpoint-400"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3math/checkpoint-400"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3math/checkpoint-1000"
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_olmo3math/checkpoint-1400"
)

# 任务列表
task_sets=(
"aime24"
"mmlu"
"gpqa_main_zeroshot"
"ifeval"
)
# ===========================================

sanitize_filename() {
    echo "$1" | tr '/' '_' | tr ',' '-'
}

echo "========================================================"
echo "Starting Evaluation using Backend: $BACKEND"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# 循环模型和任务
for model in "${models[@]}"; do
    for tasks in "${task_sets[@]}"; do
        
        # 处理文件名和日志路径
        sanitized_model=$(sanitize_filename "$model")
        sanitized_tasks=$(sanitize_filename "$tasks")
        log_file="${LOG_DIR}/log.${sanitized_model}_${sanitized_tasks}.txt"

        echo "--------------------------------------------------------"
        echo "Processing Model: $model"
        echo "Tasks: $tasks"
        
        # 检查日志是否存在，避免重复跑
        if [ -f "$log_file" ]; then
            echo "⚠️  Log file exists: $log_file"
            echo "Skipping evaluation."
            continue
        fi

        echo "Log file will be saved to: $log_file"
        echo "Running evaluation..."

        # 根据后端构建 model_args
        if [ "$BACKEND" == "vllm" ]; then
            # vLLM 针对双卡 H200 的极致优化配置
            # tensor_parallel_size=2: 让两块卡并行跑一个模型
            # gpu_memory_utilization=0.9: 预留一点显存防止炸，如果 OOM 可以调低到 0.85
            MODEL_ARGS="pretrained=$model,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1"
            BATCH_SIZE="auto"
        else
            # HuggingFace 后端 (备用)
            # parallelize=True: 让 accelerate 自动分配模型到多卡
            MODEL_ARGS="pretrained=$model,parallelize=True,dtype=auto"
            BATCH_SIZE="auto" # 也可以尝试手动设为 16 或 32
        fi

        # === 核心执行命令 ===
        # 1. 2>&1 | tee "$log_file": 让你既能看到进度条，又能保存日志
        # 2. --trust_remote_code: 很多新模型需要这个
        python -m lm_eval \
            --model "$BACKEND" \
            --model_args "$MODEL_ARGS" \
            --tasks "$tasks" \
            --batch_size "$BATCH_SIZE" \
            --apply_chat_template \
            --log_samples \
            --trust_remote_code \
	    --gen_kwargs until=["<|endoftext|>"] \
            --output_path "$OUTPUT_PATH" \
            2>&1 | tee "$log_file"

        # 获取 Python 命令的退出状态 (PIPESTATUS[0] 拿到的是 python 的状态，不是 tee 的)
        EXIT_CODE=${PIPESTATUS[0]}

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Evaluation complete for $sanitized_model"
        else
            echo "❌ Error occurred! Check log: $log_file"
        fi

    done
done

echo "--------------------------------------------------------"
echo "All evaluations completed."
