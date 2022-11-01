#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --job-name=train_reranker

#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1

train_data_path="./data/prepared/cnndm/val/dataset_train.jsonl"
dev_data_path="./data/prepared/cnndm/val/dataset_val_min.jsonl"
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
#     --run_name "debug_localize_inter" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 30 \
#     --candidate_model "pegasus_cnndm" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --localize True \
#     --localize_ratio 0.4 \
#     --sub_sampling_ratio 0.1 \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --load_checkpoint "./outputs/scr/roberta-large/basic_beam_30/checkpoint-1930" \


# # # dual BCE
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "dual" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "debug_localize" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidate 30 \
#     --candidate_model "pegasus_cnndm" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 512 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 16 \
#     --num_train_epochs 2 \
#     --overwrite_output_dir True \
#     --loss_type "triplet_simcls" \
#     --sub_sampling_mode "uniform" \
#     --sub_sampling_ratio "0.3" \
#     --num_pos 2 \
#     --num_neg 2 \
#     --localize True \
#     --localize_ratio 0.3 \


# # dual ListMLE
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
#     --n_candidate 30 \
#     --candidate_model "pegasus_cnndm" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 512 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 3 \
#     --overwrite_output_dir True \
#     --loss_type "p_ListMLE" \
#     --sub_sampling_mode "uniform" \
#     --sub_sampling_ratio 0.3


torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "debug_mse" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidate 30 \
    --candidate_model "pegasus_cnndm" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --source_maxlength 256 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --sub_sampling_mode "random" \
    --sub_sampling_ratio 0.5 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 100 \
    --evaluate_first_step True \
    --load_checkpoint "./outputs/crosscompare/roberta-large/debug_mse_BCE/checkpoint-100" \






