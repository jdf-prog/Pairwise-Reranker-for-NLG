#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1

######################################################
# Generate candidates using public fine-tuned models
######################################################

######################################################
# cnndm
######################################################

# for set in "train" "val" "test"
# do
# for method in "diverse_beam_search" "beam_search"
# do
# python generate_candidate.py \
# --dataset cnndm \
# --model_type pegasus \
# --model google/pegasus-cnn_dailymail \
# --model_name pegasus_cnndm \
# --load_model False \
# --set ${set} \
# --inference_bs 2 \
# --save_candidates True \
# --generation_method ${method} \
# --num_return_sequences 15 \
# --num_beams 15 \
# --num_beam_groups 15 \
# done
# done

######################################################
# xsum
######################################################
# for set in "train" "val" "test"
# do
#     for method in "diverse_beam_search" "beam_search"
#     do
#         python generate_candidate.py \
#         --dataset xsum \
#         --model_type pegasus \
#         --model google/pegasus-xsum \
#         --model_name pegasus_xsum \
#         --load_model False \
#         --set ${set} \
#         --inference_bs 2 \
#         --save_candidates True \
#         --generation_method ${method} \
#         --num_return_sequences 15 \
#         --num_beams 15 \
#         --num_beam_groups 15
#     done
# done

######################################################
# wmt18
######################################################

# for set in "train" "val" "test"
# do
#     for method in "beam_search" "diverse_beam_search"
#     do
#         python generate_candidate.py \
#         --dataset wmt18 \
#         --model_type opus-mt \
#         --model Helsinki-NLP/opus-mt-zh-en \
#         --model_name opus_mt \
#         --load_model False \
#         --set ${set} \
#         --inference_bs 15 \
#         --save_candidates True \
#         --generation_method ${method} \
#         --num_return_sequences 15 \
#         --num_beams 15 \
#         --num_beam_groups 15

#     done
# done

######################################################
# commongen
######################################################

# # commongen
# for set in "train" "val" "test"
# do
#     for method in "diverse_beam_search" "beam_search"
#     do
#
#         python generate_candidate.py \
#         --dataset commongen \
#         --model_type t5 \
#         --model mrm8488/t5-base-finetuned-common_gen \
#         --model_name t5_common_gen \
#         --load_model False \
#         --set ${set} \
#         --inference_bs 16 \
#         --save_candidates True \
#         --generation_method ${method} \
#         --num_return_sequences 15 \
#         --num_beams 15 \
#         --num_beam_groups 15

#     done
# done

#######################################################
# Generate candidates using our half-fine-tuned models
# these models are trained on 1/2 of the training data
# See scripts ./train_baseline.sh for more details
# These candidates are used for training the reranker
#######################################################

######################################################
# For commongen
# Generate candidates on 2-half of the training data
# with model trained on 1-half of the training data
######################################################

# for method in "diverse_beam_search" "beam_search"
# do
#     python generate_candidate.py \
#     --dataset commongen \
#     --model_type t5 \
#     --model t5-large \
#     --model_name t5_common_gen_half \
#     --load_model True \
#     --load_model_path "../../models/t5_common_gen_1_half/checkpoint-best" \
#     --partition '2_half' \
#     --set 'train' \
#     --inference_bs 10 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15
# done

######################################################
# For commongen
# Generate candidates on 1-half of the training data
# with model trained on 2-half of the training data
######################################################

# for method in "diverse_beam_search" "beam_search"
# do
#     python generate_candidate.py \
#     --dataset commongen \
#     --model_type t5 \
#     --model t5-large \
#     --model_name t5_common_gen_half \
#     --load_model True \
#     --load_model_path "../../models/t5_common_gen_2_half/checkpoint-best" \
#     --partition '1_half' \
#     --set 'train' \
#     --inference_bs 10 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15
# done

######################################################
# For cnndm
# Generate candidates on 2-half of the training data
# with model trained on 1-half of the training data
######################################################

# for method in "diverse_beam_search" "beam_search"
# do
#     python generate_candidate.py \
#     --dataset cnndm \
#     --model_type pegasus \
#     --model google/pegasus-large \
#     --model_name "pegasus_cnndm_half" \
#     --load_model True \
#     --load_model_path "../../models/pegasus_cnndm_1_half/checkpoint-best" \
#     --partition '2_half' \
#     --set train \
#     --inference_bs 16 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15 \
# done

######################################################
# For cnndm
# Generate candidates on 1-half of the training data
# with model trained on 2-half of the training data
######################################################

# for method in "diverse_beam_search" "beam_search"
# do
#     python generate_candidate.py \
#     --dataset cnndm \
#     --model_type pegasus \
#     --model google/pegasus-large \
#     --model_name "pegasus_cnndm_half" \
#     --load_model True \
#     --load_model_path "../../models/pegasus_cnndm_2_half/checkpoint-best" \
#     --partition '1_half' \
#     --set train \
#     --inference_bs 16 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15 \
# done
