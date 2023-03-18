#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH -n 1

train_data_path="./data/prepared/cnndm/val/dataset_train.jsonl"
dev_data_path="./data/prepared/cnndm/val/dataset_val_min.jsonl"
test_data_path="./data/prepared/cnndm/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../

torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_fid.py \
    --fid_type "dualfid" \
    --model_type "bart" \
    --model_name "facebook/bart-large" \
    --run_name "debug" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 1 \
    --candidate_model "pegasus_cnndm" \
    --candidate_generation_method "diverse_beam_search,beam_search" \
    --source_maxlength 384 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --learning_rate 1e-5 \
    --do_train True \
    --do_eval True \
    --do_predict False \
