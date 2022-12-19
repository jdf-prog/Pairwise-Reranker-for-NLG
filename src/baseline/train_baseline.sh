#!/bin/bash
#SBATCH --time=48:00:00
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
#     --nproc_per_node 1 \
# train_baseline.py \
#     --dataset cnndm \
#     --model_type pegasus \
#     --model google/pegasus-large \
#     --model_name pegasus_cnndm_2_half \

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

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_baseline.py \
#     --dataset wmt18 \
#     --model_type t5 \
#     --model t5-large \
#     --model_name t5_wmt18_1_half \

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

torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_baseline.py \
    --dataset commongen \
    --model_type t5 \
    --model t5-large \
    --model_name t5_common_gen_2_half \
