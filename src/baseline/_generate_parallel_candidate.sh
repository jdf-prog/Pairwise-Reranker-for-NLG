#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=generate_parallel_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1


dataset=$1
model_type=$2
model=$3
model_name=$4
load_model=$5
load_model_path=$6
set=$7
inference_bs=$8
generation_method=$9
num_shards=${10}
shard_size=${11}
shard_id=${12}

echo "dataset: $dataset"
echo "model_type: $model_type"
echo "model: $model"
echo "model_name: $model_name"
echo "load_model: $load_model"
echo "load_model_path: $load_model_path"
echo "set: $set"
echo "inference_bs: $inference_bs"
echo "generation_method: $generation_method"
echo "num_shards: $num_shards"
echo "shard_size: $shard_size"
echo "shard_id: $shard_id"



start_idx=$((shard_id * shard_size))
end_idx=$((start_idx + shard_size))

python generate_candidate.py \
--dataset "$dataset" \
--model_type "$model_type" \
--model "$model" \
--model_name "$model_name" \
--load_model "$load_model" \
--load_model_path "$load_model_path" \
--set "${set}" \
--inference_bs "$inference_bs" \
--save_candidates True \
--generation_method "$generation_method" \
--start_idx "$start_idx" \
--end_idx "$end_idx" \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15
