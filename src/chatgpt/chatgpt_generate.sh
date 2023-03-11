#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=chatgpt_generate
#SBATCH --output ../../jobs/%j.out

# python chatgpt.py \
#     --dataset commongen \
#     --set train \
#     --max_size 10000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset commongen \
#     --set val \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset commongen \
#     --set test \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset cnndm \
#     --set train \
#     --max_size 10000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset cnndm \
#     --set val \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset cnndm \
#     --set test \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset wmt18 \
#     --set train \
#     --max_size 10000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset wmt18 \
#     --set val \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \

# python chatgpt.py \
#     --dataset wmt18 \
#     --set test \
#     --max_size 1000 \
#     --n 10 \
#     --temperature 1.0 \
