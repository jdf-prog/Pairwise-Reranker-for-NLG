#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=train_baseline
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH -n 1

nvidia-smi

localhost=$RANDOM

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 4 \
# train_baseline.py \
#     --dataset cnndm \
#     --model_type pegasus \
#     --model google/pegasus-large \
#     --model_name pegasus_cnndm_2_half \
#     --evaluate_before_training False \
#     --source_max_length 1024 \
#     --target_max_length 128 \
#     --fp16 False \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 5e-5 \
#     --generation_num_beams 8 \
#     --num_train_epochs 10 \
#     --do_eval False \


torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_baseline.py \
    --dataset cnndm \
    --model_type bart \
    --model facebook/bart-large \
    --model_name bart_cnndm_2_half \
    --evaluate_before_training False \
    --source_max_length 512 \
    --target_max_length 128 \
    --fp16 True \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --generation_num_beams 5 \
    --num_train_epochs 10 \



# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_baseline.py \
#     --dataset xsum \
#     --model_type pegasus \
#     --model google/pegasus-large \
#     --model_name pegasus_xsum_1_half \
#     --fp16 False \
#     --evaluate_before_training False \
#     --source_max_length 512 \
#     --target_max_length 64 \
#     --fp16 False \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 64 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 5e-5 \
#     --generation_num_beams 8 \
#     --num_train_epochs 10 \

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_baseline.py \
#     --dataset xsum \
#     --model_type bart \
#     --model facebook/bart-large \
#     --model_name bart_xsum_2_half \
#     --evaluate_before_training False \
#     --source_max_length 512 \
#     --target_max_length 64 \
#     --fp16 True \
#     --per_device_train_batch_size 10 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 3e-5 \
#     --generation_num_beams 5 \
#     --num_train_epochs 10 \

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 4 \
# train_baseline.py \
#     --dataset wmt18 \
#     --model_type mt5 \
#     --model google/mt5-large \
#     --model_name mt5_wmt18_1_half \
#     --evaluate_before_training False \
#     --source_max_length 100 \
#     --target_max_length 200 \
#     --fp16 False \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 5e-5 \
#     --generation_num_beams 5 \
#     --num_train_epochs 10 \
#     --max_train_data_size 200 \
#     --max_eval_data_size 200 \

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_baseline.py \
#     --dataset commongen \
#     --model_type t5 \
#     --model t5-large \
#     --model_name t5_common_gen_1_half \
#     --do_predict False \
#     --evaluate_before_training False \
#     --source_max_length 25 \
#     --target_max_length 35 \
#     --fp16 True \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 64 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 5e-5 \
#     --generation_num_beams 5 \
#     --num_train_epochs 10 \
