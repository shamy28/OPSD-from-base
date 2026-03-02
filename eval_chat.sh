#!/bin/bash

#export LD_LIBRARY_PATH=/home/benchmarking/crickett.benchmarking/ss11-libs:$LD_LIBRARY_PATH
#export PATH=/home/u5o/yaolu.u5o/sjd_project/dev_home_lm_eval/.local/bin:$PATH
mkdir -p log_chatmodel # the log output dir

MAX_WAIT=300

# List of models

models=(
"/data/users/zhenyugao/sjd/self_distlitation/Self-Distillation/output_sdft_50k/checkpoint-1000"
        )

task_sets=(
"aime24"
	)



sanitize_filename() {
    echo "$1" | tr '/' '_' | tr ',' '-'
}

# Loop through each model and task set
first_run=true

for model in "${models[@]}"; do
    for tasks in "${task_sets[@]}"; do
        echo "Evaluating model: $model"
        echo "Tasks: $tasks"

                if ! $first_run; then
            echo "Waiting for previous job to complete..."
                        wait
        else
            first_run=false
        fi

        # Sanitize model name and tasks for the log file name
        sanitized_model=$(sanitize_filename "$model")
        sanitized_tasks=$(sanitize_filename "$tasks")
        log_file="log_chatmodel/log.${sanitized_model}_${sanitized_tasks}"

                # Check if log file already exists
        if [ -f "$log_file" ]; then
            echo "Log file $log_file already exists. Skipping evaluation."
            continue
        fi


        # Run the evaluation command and capture both stdout and stderr
         if output=$(CUDA_VISIBLE_DEVICES="0,1" python -m lm_eval\
                        --model vllm \
            --model_args pretrained="$model" \
            --tasks "$tasks" \
            --batch_size 4 \
            --num_fewshot 0 \
	    --verbosity DEBUG \
	    --apply_chat_template \
            --log_samples \
                        --trust_remote_code \
            --output_path mt_chatmodel 2>&1); then

            # If the command was successful, save the output to the log file
            echo "$output" > "$log_file"
            echo "Evaluation complete for $model with tasks: $tasks"
            echo "Log file: $log_file"
        else
            # If there was an error, print the error message and continue to the next iteration
            echo "Error occurred while evaluating $model with tasks: $tasks"
            echo "$output"
            echo "Skipping to next evaluation."
        fi

        echo "----------------------------------------"
    done
done

echo "All evaluations completed."
