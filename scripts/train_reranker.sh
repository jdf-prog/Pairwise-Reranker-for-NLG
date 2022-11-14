#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --nodelist=ink-molly
#SBATCH --gres=gpu:1
#SBATCH -n 1


dataset="cnndm"

train_data_path="./data/prepared/${dataset}/val/dataset_train.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset_val_min.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

# train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
# dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
# test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../

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
#     --n_candidates 30 \
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
#     --n_candidates 30 \
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
#     --n_candidates 30 \
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


# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "dualcompare" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "debug_4_pos_4_neg_mean" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 256 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --num_train_epochs 4 \
#     --overwrite_output_dir True \
#     --num_pos 4 \
#     --num_neg 4 \
#     --evaluate_first_step True \
#     # --evaluation_strategy "steps" \
#     # --save_strategy "steps" \
#     # --eval_steps 10 \
#     # --save_steps 10 \
#     # --load_checkpoint "./outputs/crosscompare/roberta-large/debug_mean_BCE/checkpoint-100" \


torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "debug" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "pegasus_cnndm" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --source_maxlength 256 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --overwrite_output_dir True \
    --num_pos 4 \
    --num_neg 4 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 100 \
    # --evaluate_first_step True \
    # --load_checkpoint "./outputs/crosscompare/roberta-large/debug_2_pos_2_neg_basic/checkpoint-best" \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \
    # --do_train False \
    # --do_eval False \
    # --do_predict True \


# # curriculum learning

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "debug_curriculum" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 256 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 3 \
#     --overwrite_output_dir True \
#     --num_pos 2 \
#     --num_neg 2 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --curriculum_learning True \
#     --curriculum_size 5000 \
#     --num_curriculum 50 \
#     --load_checkpoint "./outputs/crosscompare/roberta-large/debug_2_pos_2_neg_basic/checkpoint-best" \
#     --evaluate_first_step True \
#     --evaluation_strategy "steps" \
#     --save_strategy "steps" \
#     --eval_steps 1000 \
#     --save_steps 1000 \
#     # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_2_pos_2_neg_basic/checkpoint-3800" \
#     # --do_train False \
#     # --do_eval False \
#     # --do_predict True \


