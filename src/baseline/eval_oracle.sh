#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_oracle
#SBATCH --output ../../jobs/%j.out
#SBATCH --nodelist=ink-ellie

python eval_oracle.py \
    --dataset cnndm_bart \
    --set val \
    --metrics "rouge" \
    --num_workers 1 \
    --save_prepared True \

# python eval_oracle.py \
#     --dataset wmt18 \
#     --set train \
#     --metrics "bleu" \
#     --num_workers 1 \
#     --save_prepared True \

# do not save train prepared file, not consistent with raw dataset
# python eval_oracle.py \
#     --dataset commongen \
#     --set train \
#     --metrics "bleu,bleu4,cider,spice" \
#     --num_workers 1 \


# FOR: GPT3 transfer
# python eval_oracle.py \
#     --dataset commongen_gpt3 \
#     --set val \
#     --metrics "bleu,cider" \
#     --num_workers 1 \
#     --save_prepared True \

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


# # FOR: chatgpt

# for set in train val test; do
#     python eval_oracle.py \
#         --dataset cnndm_chatgpt \
#         --set $set \
#         --metrics "rouge" \
#         --num_workers 1 \
#         --save_prepared True \

# done

# for set in train val test; do
#     python eval_oracle.py \
#         --dataset commongen_chatgpt \
#         --set $set \
#         --metrics "bleu,cider" \
#         --num_workers 1 \
#         --save_prepared True \

# done

# for set in train val test; do
#     python eval_oracle.py \
#         --dataset wmt18_chatgpt \
#         --set $set \
#         --metrics "bleu" \
#         --num_workers 1 \
#         --save_prepared True \

# done
