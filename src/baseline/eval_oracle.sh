#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-titan
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=4

python eval_oracle.py \
    --dataset=cnndm \
    --set=val \
    --eval_rouge=True \
    --eval_bleu=True \
