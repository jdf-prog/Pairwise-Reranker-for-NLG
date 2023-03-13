#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=train_reranker_wmt18
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:8000:1
#SBATCH -n 1


dataset="wmt18_chatgpt"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"
candidate_model="gpt-3.5-turbo" # or "opus-mt"
candidate_generation_method="top_p_sampling"
n_candidates=10
learning_rate=1e-5
source_maxlength=112
candidate_maxlength=200
per_device_train_batch_size=16
per_device_eval_batch_size=256
gradient_accumulation_steps=4
num_train_epochs=5

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
    --model_type "deberta" \
    --model_name "microsoft/deberta-v3-large" \
    --run_name "train_wmt18_PairReranker" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates ${n_candidates} \
    --candidate_model ${candidate_model} \
    --candidate_generation_method ${candidate_generation_method} \
    --learning_rate ${learning_rate} \
    --source_maxlength ${source_maxlength} \
    --candidate_maxlength ${candidate_maxlength} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --reduce_type  "linear" \
    --pooling_type "special" \
    --max_train_data_size -1 \
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --using_metrics "bleu" \
    --overwrite_output_dir True \
    # --do_train False \
    # --do_eval False \
    # --do_predict True \
    # --load_checkpoint "outputs/crosscompare/microsoft/deberta-v3-large/train_wmt18_rank_based/checkpoint-best" \
    # --inference_mode 'full' \
    # --reset_scores False \
    # --evaluate_before_training True \
    # --do_predict False \
    # --load_checkpoint "./outputs/crosscompare/roberta-large/debug_2_pos_2_neg_basic/checkpoint-best" \
    # --evaluation_strategy "steps" \
    # --save_strategy "steps" \
    # --eval_steps 100 \
    # --save_steps 100 \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \


torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "scr" \
    --model_type "deberta" \
    --model_name "microsoft/deberta-v3-large" \
    --run_name "train_wmt18_SummaReranker" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates ${n_candidates} \
    --candidate_model ${candidate_model} \
    --candidate_generation_method ${candidate_generation_method} \
    --learning_rate ${learning_rate} \
    --source_maxlength ${source_maxlength} \
    --candidate_maxlength ${candidate_maxlength} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${num_train_epochs} \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "MoE_BCE" \
    --sub_sampling_mode "top_bottom" \
    --using_metrics "bleu" \
    --overwrite_output_dir True \
    # --do_train False \
    # --do_eval False \
    # --do_predict True \
    # --load_checkpoint "outputs/scr/microsoft/deberta-v3-large/train_wmt18_SummaReranker/checkpoint-best" \
    # --evaluate_before_training True \
