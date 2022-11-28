#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1


dataset="cnndm"

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
    --run_name "trian_cnndm_special" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --source_maxlength 256 \
    --candidate_maxlength 128 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "MSE" \
    --sub_sampling_mode "top_bottom" \
    --sub_sampling_ratio 0.4 \
    --max_train_data_size 50000 \
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --using_metrics "rouge1+rouge2+rougeLsum" \
    # --do_train False \
    # --do_eval False \
    # --do_predict True \
    # --load_checkpoint "./outputs/crosscompare/roberta-large/trian_cnndm_curriculum_error-based_MSE/checkpoint-best" \
    # --evaluate_before_training True \
    # --evaluation_strategy "steps" \
    # --save_strategy "steps" \
    # --eval_steps 100 \
    # --save_steps 100 \
    # --evaluate_before_training True \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \



# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "scr" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "trian_cnndm_MoE_BCE" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --loss_type "MoE_BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --num_pos 1 \
#     --num_neg 1 \
#     --learning_rate 1e-5 \
#     --max_train_data_size 50000 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     # --evaluate_before_training True \
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
#     --run_name "trian_cnndm_ranknet" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 384 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --num_train_epochs 5 \
#     --overwrite_output_dir True \
#     --loss_type "ranknet" \
#     --sub_sampling_mode "uniform" \
#     --sub_sampling_ratio 0.4 \
#     --learning_rate 1e-5 \
#     --max_train_data_size 50000 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --do_train False \
#     --do_eval False \
#     --load_checkpoint "./outputs/scr/roberta-large/trian_cnndm_ranknet/checkpoint-3125" \
#     # --evaluate_before_training True \


# # self-adapted curriculum
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "trian_cnndm_curriculum_larger_batch_size" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 256 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 10 \
#     --overwrite_output_dir True \
#     --num_pos 1 \
#     --num_neg 1 \
#     --loss_type "BCE" \
#     --sub_sampling_mode "top_bottom" \
#     --max_train_data_size 50000 \
#     --max_eval_data_size 200 \
#     --max_predict_data_size -1 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --load_checkpoint "./outputs/crosscompare/roberta-large/trian_cnndm/checkpoint-9375" \
#     --curriculum_size 50000 \
#     --evaluate_before_training True \
#     --evaluation_strategy "steps" \
#     --save_strategy "steps" \
#     --eval_steps 1000 \
#     --save_steps 1000 \
#     --curriculum_learning "self-adapted" \
#     --learning_rate 1e-6 \
#     # --evaluation_strategy "steps" \
#     # --save_strategy "steps" \
#     # --eval_steps 100 \
#     # --save_steps 100 \
#     # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \
#     # --do_train False \
#     # --do_eval False \
#     # --do_predict True \

# # easy2hard or error-based curriculum
# train_data_path="./data/prepared/${dataset}/train/dataset.50000_100000.jsonl"
# curriculum_indices_path="./data/prepared/${dataset}/train/error_indices.50000_100000.npy"
# torchrun \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint="localhost:${localhost}" \
#     --nnodes 1 \
#     --nproc_per_node 1 \
# train_reranker.py \
#     --reranker_type "crosscompare" \
#     --model_type "roberta" \
#     --model_name "roberta-large" \
#     --run_name "trian_cnndm_curriculum_error-based_MSE" \
#     --train_data_path ${train_data_path} \
#     --eval_data_path ${dev_data_path} \
#     --test_data_path ${test_data_path} \
#     --n_candidates 30 \
#     --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
#     --candidate_generation_method "diverse_beam_search+beam_search" \
#     --source_maxlength 256 \
#     --candidate_maxlength 128 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 32 \
#     --num_train_epochs 2 \
#     --overwrite_output_dir True \
#     --loss_type "MSE" \
#     --max_train_data_size 300000 \
#     --max_eval_data_size -1 \
#     --max_predict_data_size 1000 \
#     --using_metrics "rouge1+rouge2+rougeLsum" \
#     --curriculum_learning "error-based" \
#     --curriculum_indices_path ${curriculum_indices_path} \
#     # --do_train False \
#     # --do_eval False \
#     # --do_predict True \
#     # --evaluate_before_training True \
#     # --load_checkpoint "./outputs/crosscompare/roberta-large/trian_cnndm/checkpoint-9375" \
#     # --sub_sampling_mode "top_bottom" \
#     # --evaluation_strategy "steps" \
#     # --save_strategy "steps" \
#     # --eval_steps 100 \
#     # --save_steps 100 \
#     # --evaluate_before_training True \
#     # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \

