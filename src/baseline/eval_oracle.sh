#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=bash
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=1

python eval_oracle.py \
    --dataset wmt18 \
    --set test \
    --eval_rouge False \
    --eval_bleu True \
    --num_workers 1 \
