#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=bash
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=1

# python eval_oracle.py \
#     --dataset cnndm \
#     --set test \
#     --metrics "rouge" \
#     --num_workers 1 \

# python eval_oracle.py \
#     --dataset xsum \
#     --set test \
#     --metrics "rouge" \
#     --num_workers 2 \

python eval_oracle.py \
    --dataset wmt18 \
    --set test \
    --metrics "bleu" \
    --num_workers 1 \

# python eval_oracle.py \
#     --dataset commongen \
#     --set val \
#     --metrics "bleu,cider" \
#     --num_workers 1 \
