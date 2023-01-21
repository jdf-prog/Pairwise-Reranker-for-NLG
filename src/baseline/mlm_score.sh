#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=mlm_score
#SBATCH --output ../../jobs/%j.out
#SBATCH --nodelist=ink-ron
#SBATCH --gres=gpu:2
#SBATCH --qos=general
#SBATCH -n 1

nvidia-smi

dataset="cnndm_gpt3"
set="test"
generation_model="gpt3"
generation_method="top_p_sampling"

candidate_file="../../data/${dataset}/${set}/${generation_method}/candidates_${generation_model}.pkl"
save_path="../../data/${dataset}/${set}/${generation_method}/cand_scores_${generation_model}_mlm.pkl"

python mlm_score.py \
    --model roberta-large-en-cased \
    --n_gpu 2 \
    --candidate_file ${candidate_file} \
    --save_path ${save_path} \
