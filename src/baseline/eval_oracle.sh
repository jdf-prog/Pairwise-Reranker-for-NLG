#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=eval_oracle
#SBATCH --output ../../jobs/%j.out
#SBATCH --cpus-per-task=4

python eval_oracle.py \
    --dataset cnndm \
    --set test \
    --metrics "rouge" \
    --num_workers 4 \

# python eval_oracle.py \
#     --dataset xsum \
#     --set test \
#     --metrics "rouge" \
#     --num_workers 1 \
#     --save_prepared True \

# python eval_oracle.py \
#     --dataset wmt18 \
#     --set test \
#     --metrics "bleu" \
#     --num_workers 1 \

# python eval_oracle.py \
#     --dataset commongen \
#     --set val \
#     --metrics "bleu,cider" \
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
