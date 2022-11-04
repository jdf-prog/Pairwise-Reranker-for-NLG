#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=FID_train
#SBATCH --nodelist=ink-titan
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1
CUDA_LAUNCH_BLOCKING=1 # for debugging
NGPU=1

train_data_path="./data/cnn_dailymail_train_hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo_min.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"

model_type='bart'
model_size="large"
name="diverse_beam_search"
checkpoint_dir="checkpoint/${model_type}-${model_size}"
source_maxlength=1024
candidate_maxlength=128

echo "model type: ${model_type}"
echo "model size: ${model_size}"
echo "name: ${name}"
echo "source_maxlength: ${source_maxlength}"
echo "candidate_maxlength: ${candidate_maxlength}"

nvidia-smi
# torchrun --nproc_per_node=$NGPU \
python \
        train_model.py \
        --name "${name}" \
        --train_data ${train_data_path} \
        --eval_data ${dev_data_path} \
        --model_type ${model_type} \
        --model_size ${model_size} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --checkpoint_dir ${checkpoint_dir} \
        --lr 0.00003 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.001 \
        --per_gpu_batch_size 1 \
        --n_candidate 15 \
        --total_step 25000 \
        --warmup_step 3000 \
        --main_port 19004 \
        --accumulation_steps 4 \
