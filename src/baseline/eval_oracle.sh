#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-titan
#SBATCH --output ../../jobs/%j.out
#SBATCH -n 1

python eval_oracle.py \
    --dataset=cnndm \
    --set=test \
    --eval_rouge=True \
    --eval_bleu=True \
