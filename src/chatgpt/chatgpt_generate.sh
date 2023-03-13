#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=chatgpt_generate
#SBATCH --output ../../jobs/%j.out

dataset="cnndm"                   # dataset to use
set="train"                        # set to use
generation_method="num_in_prompt" # method for generation
temperature=1.0                   # temperature for generation sampling
n=10                              # number of generations
do_eval="True"                   # whether to evaluate the generations

# Since some splits of the datasets are too large, we will load the shuffled ones
# We use max_size to control the size of the shuffled dataset
# We use the shuffled ones for
# 1. cnndm train (30000), val (3000), full
# 2. wmt18 train (50000)
# Commongen is small enough to be fully generated
load_shuffle="True"              # whether to load the shuffled dataset
max_size=30000                   # maximum size of the dataset, set to -1 to use the whole dataset

if [ "$do_eval" = "True" ]; then

    if [ "$dataset" = "cnndm" ]; then
        eval_metrics="rouge1,rouge2,rougeL,rougeLsum"
    elif [ "$dataset" = "wmt18" ]; then
        eval_metrics="bleu"
    elif [ "$dataset" = "commongen" ]; then
        eval_metrics="bleu,cider"
    fi
    save_scores="False"
else
    eval_metrics=""
    save_scores="False"
fi

python chatgpt.py \
    --dataset $dataset \
    --set $set \
    --n $n \
    --generation_method $generation_method \
    --temperature $temperature \
    --eval_metrics $eval_metrics \
    --save_scores $save_scores \
    --load_shuffle $load_shuffle \
    --max_size $max_size \
