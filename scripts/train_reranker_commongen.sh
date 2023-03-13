#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=train_reranker_commongen
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:2080:1

# about 9 hours per training epoch
# quick for each dev evaluation

dataset="commongen_chatgpt"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"
test_data_path=$dev_data_path # since commongen do not release test set, we use dev set as test set
candidate_model="gpt-3.5-turbo" # or "t5_common_gen_half+t5_common_gen"
candidate_generation_method="top_p_sampling"
n_candidates=10
learning_rate=1e-5
source_maxlength=25
candidate_maxlength=35
per_device_train_batch_size=16
per_device_eval_batch_size=128
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
    --run_name "train_${dataset}_PairReranker" \
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
    --using_metrics "bleu+cider" \
    --overwrite_output_dir True \
    # --do_predict True \
    # --do_train False \
    # --do_eval False \
    # --load_checkpoint "./outputs/crosscompare/microsoft/deberta-v3-large/train_commongen_top_bottom_4_method_based/checkpoint-best" \
    # --inference_mode "full"


# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "deberta" \
#     --model_name "microsoft/deberta-v3-large" \
#     --run_name "train_${dataset}_SummaReranker" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates ${n_candidates} \
#     --candidate_model ${candidate_model} \
#     --candidate_generation_method ${candidate_generation_method} \
#     --learning_rate ${learning_rate} \
#     --source_maxlength ${source_maxlength} \
#     --candidate_maxlength ${candidate_maxlength} \
#     --per_device_train_batch_size ${per_device_train_batch_size} \
#     --per_device_eval_batch_size ${per_device_eval_batch_size} \
#     --gradient_accumulation_steps ${gradient_accumulation_steps} \
#     --num_train_epochs ${num_train_epochs} \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --using_metrics "bleu+cider" \
#     --overwrite_output_dir True \
#     # --do_train False \
#     # --do_eval False \
#     # --do_predict True \
#     # --load_checkpoint "./outputs/scr/microsoft/deberta-v3-large/train_commongen_SummaReranker/checkpoint-best" \
