#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-gary
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/cnn_dailymail_train_hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"
model_type='bart'
model_size="large"
name="pure_bart-large"

checkpoint_dir="checkpoint/${model_type}-${model_size}"
model_path="${checkpoint_dir}/${name}/checkpoint/best_dev"
source_maxlength=512
candidate_maxlength=200

echo "model type: ${model_type}"
echo "model size: ${model_size}"
echo "name: ${name}"
echo "source_maxlength: ${source_maxlength}"
echo "candidate_maxlength: ${candidate_maxlength}"

nvidia-smi
python test_model.py \
        --model_type ${model_type} \
        --model_size ${model_size} \
        --name ${name} \
        --checkpoint_dir ${checkpoint_dir} \
        --model_path ${model_path} \
        --eval_data ${test_data_path} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_gpu_batch_size 2 \
        --n_candidate 6 \
        --main_port 19001 \
