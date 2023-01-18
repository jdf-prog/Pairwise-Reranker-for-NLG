#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1

nvidia-smi
######################################################
# Generate candidates using public fine-tuned models
######################################################

######################################################
# cnndm
######################################################

# for set in "test"
# do
#     for method in "beam_search"
#     do
#         python generate_candidate.py \
#         --dataset cnndm \
#         --model_type pegasus \
#         --model google/pegasus-large \
#         --model_name pegasus_cnndm_2_half \
#         --load_model True \
#         --set ${set} \
#         --inference_bs 2 \
#         --save_candidates True \
#         --generation_method ${method} \
#         --num_return_sequences 15 \
#         --num_beams 15 \
#         --num_beam_groups 15 \
#         --load_model_path "../../models/pegasus_cnndm_2_half/checkpoint-best" \

#     done
# done

######################################################
# xsum
######################################################
# for set in "test"
# do
#     for method in "top_p_sampling" "top_k_sampling" "beam_search" "diverse_beam_search"
#     do
#         python generate_candidate.py \
#         --dataset xsum \
#         --model_type bart \
#         --model facebook/bart-large-xsum \
#         --model_name bart_xsum \
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

######################################################
# wmt18
######################################################

# for set in "val" "test"
# do
#     for method in "top_p_sampling"
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

# commongen
for set in "val"
do
    for method in "diverse_beam_search" "top_p_sampling" "top_k_sampling"
    do

        python generate_candidate.py \
        --dataset commongen \
        --model_type t5 \
        --model t5-large \
        --model_name t5_common_gen \
        --load_model True \
        --load_model_path "../../models/t5_common_gen/checkpoint-best" \
        --set ${set} \
        --inference_bs 10 \
        --save_candidates True \
        --generation_method ${method} \
        # --num_return_sequences 1 \
        # --num_beams 1 \
        # --num_beam_groups 1 \
        # --repetition_penalty 2.5

    done
done

#######################################################
# Generate candidates using our half-fine-tuned models
# these models are trained on 1/2 of the training data
# See scripts ./train_baseline.sh for more details
# These candidates are used for training the reranker
#######################################################

######################################################
# For commongen
######################################################

# for method in "top_k_sampling" "top_p_sampling"
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
# For cnndm
######################################################

# for method in "top_p_sampling"
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
#     --inference_bs 2 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15 \

# done


######################################################
# For xsum
######################################################
# 7 inference_bs on 11GB GPU, 14 hours per decoding method
# 16 infernce_bs on 24GB GPU, 10 hours per decoding method

# for method in "top_p_sampling"
# do
#     python generate_candidate.py \
#     --dataset xsum \
#     --model_type bart \
#     --model facebook/bart-large \
#     --model_name "bart_xsum_half" \
#     --load_model True \
#     --load_model_path "../../models/bart_xsum_1_half/checkpoint-best" \
#     --partition '2_half' \
#     --set train \
#     --inference_bs 16 \
#     --save_candidates True \
#     --generation_method $method \
#     --num_return_sequences 15 \
#     --num_beams 15 \
#     --num_beam_groups 15 \
#     --source_max_length 512 \
#     --candidate_max_length 64 \

# done
