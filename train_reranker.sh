#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-titan
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/prepared/cnndm/val/dataset_train.jsonl"
dev_data_path="./data/prepared/cnndm/val/dataset_val.jsonl"
test_data_path="./data/prepared/cnndm/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "basic" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 30 \
#     --candidate_model "pegasus_cnndm" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 3 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 3 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \


# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "ListNet-basic" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 10 \
#     --candidate_model "pegasus_cnndm" \
#     --generation_method "diverse_beam_search" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 3 \
#     --overwrite_output_dir True \
#     --loss_type "ListNet"


# # dual info NCE
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "dual" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "basic" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 10 \
#     --candidate_model "pegasus_cnndm" \
#     --generation_method "diverse_beam_search" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 3 \
#     --overwrite_output_dir True \
#     --loss_type "infoNCE"
#     --num_pos 2 \
#     --num_neg 2 \

# dual ListMLE
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "dual" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "basic" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidate 60 \
    --candidate_model "pegasus_cnndm" \
    --generation_method "diverse_beam_search" \
    --source_maxlength 384 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --overwrite_output_dir True \
    --loss_type "ListMLE"




