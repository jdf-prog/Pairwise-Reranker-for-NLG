#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=bash
#SBATCH --output ../../jobs/%j.out

python eval_oracle.py \
    --dataset=cnndm \
    --set=test \
    --eval_rouge=True \
    --eval_bleu=True \
