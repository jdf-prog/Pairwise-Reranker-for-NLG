#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --nodelist=ink-ellie
#SBATCH --gres=gpu:1
#SBATCH -n 1


dataset="wmt18"

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

cd ../

# wmt18
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
train_reranker.py \
    --reranker_type "crosscompare" \
    --model_type "xlm-roberta" \
    --model_name "xlm-roberta-large" \
    --run_name "train_wmt18" \
    --train_data_path ${train_data_path} \
    --eval_data_path ${dev_data_path} \
    --test_data_path ${test_data_path} \
    --n_candidates 30 \
    --candidate_model "opus_mt" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --source_maxlength 100 \
    --candidate_maxlength 200 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 5 \
    --overwrite_output_dir True \
    --num_pos 1 \
    --num_neg 1 \
    --loss_type "BCE" \
    --sub_sampling_mode "top_bottom" \
    --max_train_data_size -1 \
    --max_eval_data_size -1 \
    --max_predict_data_size -1 \
    --using_metrics "bleu" \
    # --evaluate_before_training True \
    # --do_predict False \
    # --load_checkpoint "./outputs/crosscompare/roberta-large/debug_2_pos_2_neg_basic/checkpoint-best" \
    # --evaluation_strategy "steps" \
    # --save_strategy "steps" \
    # --eval_steps 100 \
    # --save_steps 100 \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \
    # --do_train False \
    # --do_eval False \
