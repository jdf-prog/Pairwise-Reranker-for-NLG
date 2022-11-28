#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --job-name=curriculum_inference
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1

dataset="cnndm"
# data_path="./data/prepared/${dataset}/train/dataset.50000_100000.jsonl"
data_path="./data/prepared/${dataset}/test/dataset.jsonl"

nvidia-smi

localhost=$RANDOM

num_shards=$1
shard_id=$2

cd ../
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node 1 \
curriculum_inference.py \
    --reranker_type "crosscompare" \
    --model_type "roberta" \
    --model_name "roberta-large" \
    --data_path ${data_path} \
    --n_candidates 30 \
    --candidate_model "pegasus_cnndm+pegasus_cnndm_half" \
    --candidate_generation_method "diverse_beam_search+beam_search" \
    --using_metrics "rouge1+rouge2+rougeLsum" \
    --source_maxlength 256 \
    --candidate_maxlength 128 \
    --per_device_eval_batch_size 32 \
    --overwrite_output_dir True \
    --loss_type "BCE" \
    --max_data_size 100 \
    --num_shards "$num_shards" \
    --shard_id "$shard_id" \
    --load_checkpoint "./outputs/crosscompare/roberta-large/trian_cnndm_curriculum_right-based/checkpoint-best" \
    --output_dir "./temp"
    # --load_checkpoint "./outputs/crosscompare/roberta-large/trian_cnndm/checkpoint-9375" \
    # --evaluation_strategy "steps" \
    # --save_strategy "steps" \
    # --eval_steps 100 \
    # --save_steps 100 \
    # --evaluate_before_training True \
    # --resume_from_checkpoint "./outputs/crosscompare/roberta-large/debug_poisson_dynamic/checkpoint-2000" \
