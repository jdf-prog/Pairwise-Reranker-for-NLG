#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-titan
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/prepared/cnndm/val/dataset_train.jsonl"
dev_data_path="./data/prepared/cnndm/val/dataset_val.jsonl" # debug
test_data_path="./data/prepared/cnndm/test/dataset.jsonl"

nvidia-smi

torchrun --standalone --nnodes 1 --nproc_per_node 1\
    train_reranker.py \
    --reranker_type "scr" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "basic" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --n_candidate 10 \
    --source_maxlength 384 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --max_steps 25000 \
    --overwrite_output_dir True \
