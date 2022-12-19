#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:6000:1

# about 9 hours per training epoch
# quick for each dev evaluation

dataset="commongen"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../

torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --run_name "train_commongen_single_moe" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 60 \
    --candidate_model "t5_common_gen_half+t5_common_gen" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --source_maxlength 25 \
    --candidate_maxlength 35 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --reduce_type  "single_moe" \
    --pooling_type "special" \
    --sub_sampling_ratio 0.5 \
    --max_train_data_size -1 \
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --do_predict False \
    --using_metrics "bleu+cider" \
    # --evaluate_before_training True \
    # --evaluation_strategy "steps" \
    # --save_strategy "steps" \
    # --eval_steps 5000 \
    # --save_steps 5000 \
    # --load_checkpoint "./outputs/crosscompare/roberta-large/train_commongen_error-based/checkpoint-10000" \
    # --curriculum_learning "self-adapted" \
    # --curriculum_size 100000 \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/train_commongen_debug_target_only_1/checkpoint-1000" \
    # --do_train False \
    # --do_eval False \


# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "train_commongen_debug_MoE_BCE" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "t5_common_gen_half+t5_common_gen" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 25 \
#     --candidate_maxlength 35 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 2 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --sub_sampling_ratio 1.0 \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --using_metrics "bleu+cider" \
#     --do_predict False \
#     --max_train_data_size -1 \
#     --evaluate_before_training True \
#     # --evaluation_strategy "steps" \
#     # --save_strategy "steps" \
#     # --eval_steps 1000 \
#     # --save_steps 1000 \
#     # --load_checkpoint "./outputs/scr/roberta-large/basic_beam_30/checkpoint-1930" \


# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "train_commongen_ranknet_full_data" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "t5_common_gen_half+t5_common_gen" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 25 \
#     --candidate_maxlength 35 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 1 \
#     --overwrite_output_dir True \
#     --loss_type "ranknet" \
#     --sub_sampling_mode "random" \
#     --sub_sampling_ratio 0.4 \
#     --learning_rate 1e-5 \
#     --using_metrics "bleu+cider" \
#     --do_predict False \
#     --max_train_data_size -1 \
#     --evaluate_before_training True \
#     # --load_checkpoint "./outputs/scr/roberta-large/basic_beam_30/checkpoint-1930" \
