#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=test_gpt3_cnndm
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:6000:1
# SBATCH --qos=general
#SBATCH -n 1

dataset="cnndm_gpt3"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../


# Ours deberta
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "deberta" \
    --model_name "microsoft/deberta-v3-large" \
    --run_name "test_cnndm_gpt3_PairReranker_full_comparison" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "gpt3" \
    --candidate_generation_method "top_p_sampling" \
    --source_maxlength 256 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --reduce_type  "single_linear" \
    --pooling_type "special" \
    --sub_sampling_ratio 0.1 \
    --max_train_data_size -1 \
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --using_metrics "rouge1+rouge2+rougeLsum" \
    --load_checkpoint "outputs/crosscompare/microsoft/deberta-v3-large/trian_cnndm_BCE_single_linear/checkpoint-best" \
    --inference_mode "full" \
    --do_train False \
    --do_eval False \
    --do_predict True \

# # Ours Roberta
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "test_cnndm_gpt3_PairReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 256 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 128 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --reduce_type  "single_linear" \
#     --pooling_type "special" \
#     --sub_sampling_ratio 0.1 \
#     --max_train_data_size -1 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --load_checkpoint "outputs/crosscompare/roberta-large/trian_cnndm_BCE_single_linear/checkpoint-best" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \


# # SummaReranker

# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "test_cnndm_gpt3_SummaReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 1 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --max_train_data_size -1 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --load_checkpoint "./outputs/scr/roberta-large/train_cnndm_MoE_BCE/checkpoint-best" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     # --evaluate_before_training True \


# # SummaReranker Deberta
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "deberta" \
#     --model_name "microsoft/deberta-v3-large" \
#     --run_name "test_cnndm_gpt3_SummaReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --num_train_epochs 1 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --max_train_data_size -1 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --load_checkpoint "./outputs/scr/microsoft/deberta-v3-large/train_cnndm_SummaReranker/checkpoint-best" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     # --evaluate_before_training True \
