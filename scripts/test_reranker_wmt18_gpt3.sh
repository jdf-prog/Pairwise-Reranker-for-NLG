#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=test_wmt18_gpt3
#SBATCH --output ../jobs/%j.out
#SBATCH --nodelist=ink-titan
#SBATCH --gres=gpu:1
#SBATCH -n 1


dataset="wmt18_gpt3"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../

# # Ours DeBERTa
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "deberta" \
#     --model_name "microsoft/deberta-v3-large" \
#     --run_name "test_wmt18_gpt3_PairReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 112 \
#     --candidate_maxlength 200 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 32 \
#     --gradient_accumulation_steps 2 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --reduce_type  "linear" \
#     --pooling_type "special" \
#     --using_metrics "bleu" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     --load_checkpoint "outputs/crosscompare/microsoft/deberta-v3-large/train_wmt18_linear/checkpoint-best" \

# Ours RoBERTa
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "xlm-roberta" \
    --model_name "xlm-roberta-large" \
    --run_name "test_wmt18_gpt3_PairReranker" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "gpt3" \
    --candidate_generation_method "top_p_sampling" \
    --source_maxlength 112 \
    --candidate_maxlength 200 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --reduce_type  "linear" \
    --pooling_type "special" \
    --using_metrics "bleu" \
    --do_train False \
    --do_eval False \
    --do_predict True \
    --load_checkpoint "outputs/crosscompare/xlm-roberta-large/train_wmt18_linear/checkpoint-best" \

# # SummaReranker
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "xlm-roberta" \
#     --model_name "xlm-roberta-large" \
#     --run_name "train_wmt18_SummaReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "gpt3" \
#     --candidate_generation_method "top_p_sampling" \
#     --source_maxlength 112 \
#     --candidate_maxlength 200 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --using_metrics "bleu" \
#     --do_train False \
#     --do_eval False \
#     --do_predict True \
#     --load_checkpoint "outputs/scr/xlm-roberta-large/train_wmt18_SummaReranker/checkpoint-best" \
