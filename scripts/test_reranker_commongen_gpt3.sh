#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=test_commongen_gpt3
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:6000:1

# about 9 hours per training epoch
# quick for each dev evaluation

dataset="commongen_gpt3"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

test_data_path=${dev_data_path} # commongen's test set is not public

nvidia-smi

localhost=$RANDOM

cd ../

# # Ours Deberta
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "deberta" \
#     --model_name "microsoft/deberta-v3-large" \
#     --run_name "test_commongen_gpt3_PairReranker_full_comparison" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 25 \
#     --candidate_maxlength 35 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --reduce_type  "linear" \
#     --pooling_type "special" \
#     --sub_sampling_ratio 0.5 \
#     --max_train_data_size -1 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --do_predict False \
#     --using_metrics "bleu+cider" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     --inference_mode 'full' \
#     --load_checkpoint "outputs/crosscompare/microsoft/deberta-v3-large/train_commongen_linear/checkpoint-best" \

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
#     --run_name "test_commongen_gpt3_PairReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 25 \
#     --candidate_maxlength 35 \
#     --per_device_train_batch_size 64 \
#     --per_device_eval_batch_size 64 \
#     --gradient_accumulation_steps 1 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --reduce_type  "linear" \
#     --pooling_type "special" \
#     --sub_sampling_ratio 0.5 \
#     --max_train_data_size -1 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --do_predict False \
#     --using_metrics "bleu+cider" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     --load_checkpoint "outputs/crosscompare/roberta-large/train_commongen_linear/checkpoint-best" \

# # SummaReranker Roberta
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "test_commongen_gpt3_SummaReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 25 \
#     --candidate_maxlength 35 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
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
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     --load_checkpoint "outputs/scr/roberta-large/train_commongen_SummaReranker/checkpoint-best" \

# Summareranker Deberta
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "scr" \
    --model_type "deberta" \
    --model_name "microsoft/deberta-v3-large" \
    --run_name "test_commongen_gpt3_SummaReranker" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "gpt3" \
    --candidate_generation_method "top_p_sampling" \
    --source_maxlength 25 \
    --candidate_maxlength 35 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --loss_type "MoE_BCE" \
    --sub_sampling_mode "top_bottom" \
    --sub_sampling_ratio 1.0 \
    --num_pos 1 \
    --num_neg 1 \
    --learning_rate 1e-5 \
    --using_metrics "bleu+cider" \
    --do_predict False \
    --max_train_data_size -1 \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --load_checkpoint "outputs/scr/microsoft/deberta-v3-large/train_commongen_SummaReranker/checkpoint-best" \
