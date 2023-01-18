#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=eval_oracle
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=8

# python eval_oracle.py \
#     --dataset cnndm \
#     --set train \
#     --metrics "rouge" \
#     --num_workers 1 \
#     --save_prepared True \

# python eval_oracle.py \
#     --dataset xsum \
#     --set test \
#     --metrics "rouge" \
#     --num_workers 1 \
#     --save_prepared True \

python eval_oracle.py \
    --dataset wmt18 \
    --set train \
    --metrics "bleu" \
    --num_workers 1 \
    --save_prepared True \

# do not save train prepared file, not consistent with raw dataset
# python eval_oracle.py \
#     --dataset commongen \
#     --set train \
#     --metrics "bleu,bleu4,cider,spice" \
#     --num_workers 1 \

# python eval_oracle.py \
#     --dataset commongen_gpt3 \
#     --set val \
#     --metrics "bleu,cider" \
#     --num_workers 1 \
# #     --overwrite True \
# #     --save_prepared True \

# python eval_oracle.py \
#     --dataset wmt18_gpt3 \
#     --set test \
#     --metrics "bleu" \
#     --overwrite True \
#     --save_prepared True \

# python eval_oracle.py \
#     --dataset cnndm_gpt3 \
#     --set test \
#     --metrics "rouge" \
#     --num_workers 1 \
