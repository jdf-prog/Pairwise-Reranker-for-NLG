#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=gpt3_generate
#SBATCH --output ../../jobs/%j.out

# python gpt3.py \
#     --dataset commongen \
#     --set val \

# python gpt3.py \
#     --dataset cnndm \
#     --set test \

python gpt3.py \
    --dataset wmt18 \
    --set test \
