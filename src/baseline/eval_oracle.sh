#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=bash
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=4

python eval_oracle.py \
    --dataset cnndm \
    --set test \
    --eval_rouge True \
    --eval_bleu True \
    --num_workers 4 \
