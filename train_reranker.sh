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

# torchrun --standalone --nnodes 1 --nproc_per_node 1 --master_port 20000 \
#     train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "basic" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 30 \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --load_checkpoint ./outputs/scr/roberta-large/basic/checkpoint-3000 \
#     --do_train False \
#     --do_eval False \


torchrun --standalone --nnodes 1 --nproc_per_node 1 --master_port 20001 \
    train_reranker.py \
    --reranker_type "dual" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "4_neg" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidate 30 \
    --source_maxlength 384 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 4 \
