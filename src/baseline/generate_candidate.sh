#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-gary
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

python generate_candidate.py \
--dataset cnndm \
--model_type bart \
--model facebook/bart-large-cnn \
--model_name bart_cnndm \
--load_model False \
--set val \
--inference_bs 2 \
--save_candidates True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

python generate_candidate.py \
--dataset cnndm \
--model_type pegasus \
--model google/pegasus-cnn_dailymail \
--model_name pegasus_cnndm \
--load_model False \
--set val \
--inference_bs 2 \
--save_candidates True \
--generation_method diverse_beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

python generate_candidate.py \
--dataset cnndm \
--model_type bart \
--model facebook/bart-large-cnn \
--model_name bart_cnndm \
--load_model False \
--set val \
--inference_bs 2 \
--save_candidates True \
--generation_method beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \

python generate_candidate.py \
--dataset cnndm \
--model_type pegasus \
--model google/pegasus-cnn_dailymail \
--model_name pegasus_cnndm \
--load_model False \
--set val \
--inference_bs 2 \
--save_candidates True \
--generation_method beam_search \
--num_return_sequences 15 \
--num_beams 15 \
--num_beam_groups 15 \
