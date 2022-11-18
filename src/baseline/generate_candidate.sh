#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH -n 1

# python generate_candidate.py \
# --dataset cnndm \
# --model_type bart \
# --model facebook/bart-large-cnn \
# --model_name bart_cnndm \
# --load_model False \
# --set train \
# --inference_bs 2 \
# --save_candidates True \
# --generation_method top_p_sampling \
# --num_return_sequences 15 \
# --num_beams 15 \
# --num_beam_groups 15 \

# python generate_candidate.py \
# --dataset cnndm \
# --model_type pegasus \
# --model google/pegasus-cnn_dailymail \
# --model_name pegasus_cnndm \
# --load_model False \
# --set train \
# --inference_bs 2 \
# --save_candidates True \
# --generation_method beam_search \
# --num_return_sequences 15 \
# --num_beams 15 \
# --num_beam_groups 15 \

# python generate_candidate.py \
# --dataset cnndm \
# --model_type bart \
# --model facebook/bart-large-cnn \
# --model_name bart_cnndm \
# --load_model False \
# --set val \
# --inference_bs 2 \
# --save_candidates True \
# --generation_method top_k_sampling \
# --num_return_sequences 15 \
# --num_beams 15 \
# --num_beam_groups 15 \

# python generate_candidate.py \
# --dataset cnndm \
# --model_type pegasus \
# --model google/pegasus-cnn_dailymail \
# --model_name pegasus_cnndm \
# --load_model False \
# --set val \
# --inference_bs 2 \
# --save_candidates True \
# --generation_method top_k_sampling \
# --num_return_sequences 15 \
# --num_beams 15 \
# --num_beam_groups 15 \

# cnndm
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

# xsum
# for set in "val" "train"
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

# wmt18
for set in "train"
do
    for method in "beam_search" "diverse_beam_search"
    do
        python generate_candidate.py \
        --dataset wmt18 \
        --model_type opus-mt \
        --model Helsinki-NLP/opus-mt-zh-en \
        --model_name opus_mt \
        --load_model False \
        --set ${set} \
        --inference_bs 15 \
        --save_candidates True \
        --generation_method ${method} \
        --num_return_sequences 15 \
        --num_beams 15 \
        --num_beam_groups 15

    done
done

# python generate_candidate.py \
#         --dataset xsum \
#         --model_type pegasus \
#         --model google/pegasus-xsum \
#         --model_name pegasus_xsum \
#         --load_model False \
#         --set 'train' \
#         --inference_bs 2 \
#         --save_candidates True \
#         --generation_method "beam_search" \
#         --num_return_sequences 15 \
#         --num_beams 15 \
#         --num_beam_groups 15


# # commongen
# for set in "train" "val" "test"
# do
#     for method in "diverse_beam_search" "beam_search"
#     do
#         # python generate_candidate.py \
#         # --dataset commongen \
#         # --model_type flan-t5 \
#         # --model google/flan-t5-large \
#         # --model_name flan-t5-large \
#         # --load_model False \
#         # --set ${set} \
#         # --inference_bs 16 \
#         # --save_candidates True \
#         # --generation_method ${method} \
#         # --num_return_sequences 15 \
#         # --num_beams 15 \
#         # --num_beam_groups 15

#         # python generate_candidate.py \
#         # --dataset commongen \
#         # --model_type bart \
#         # --model sibyl/BART-large-commongen \
#         # --model_name bart_common_gen \
#         # --load_model False \
#         # --set ${set} \
#         # --inference_bs 16 \
#         # --save_candidates True \
#         # --generation_method ${method} \
#         # --num_return_sequences 15 \
#         # --num_beams 15 \
#         # --num_beam_groups 15

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
