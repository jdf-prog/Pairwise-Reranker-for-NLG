#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output ../jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ink-gary

nvidia-smi
cd ../

localhost=$RANDOM
dataset="commongen"
backbone_type="roberta" # "deberta" or "roberta"
backbone_name="roberta-large" # "microsoft/deberta-v3-large" or "roberta-large"
n_gpu=1
reranker="SimCLS" # "PairReranker" or "SummaReranker" or "SimCLS"
candidate_model="t5_commongen_half,t5_common_gen"
candidate_generation_method="beam_search,diverse_beam_search"
n_candidates=30
learning_rate=1e-5
num_train_epochs=5
max_train_data_size=-1 # -1 means no limit
max_eval_data_size=-1 # -1 means no limit
max_predict_data_size=-1 # -1 means no limit
do_inference=False # whether do inference instead of training, i.e. do test
# for inference, sometimes you want to use a checkpoint trained on another dataset
# to do inference on a dataset, you can set the checkpoint_trained_dataset to the dataset
# by default, it is set to the dataset you are doing inference on
checkpoint_trained_dataset=${dataset}

train_data_path="./data/prepared/${dataset}/train/dataset.jsonl"
dev_data_path="./data/prepared/${dataset}/val/dataset.jsonl"
test_data_path="./data/prepared/${dataset}/test/dataset.jsonl"
# set the dataset specific parameters below
if [[ $dataset =~ "cnndm" ]]; then
    echo "Using cnndm"
    source_maxlength=256
    candidate_maxlength=128
    per_device_train_batch_size=4
    per_device_eval_batch_size=16
    gradient_accumulation_steps=16
    using_metrics="rouge1,rouge2,rougeLsum"
    # checkpoint_trained_dataset="trian_cnndm_BCE_single_linear" # for personal history reason

elif [[ $dataset =~ "commongen" ]]; then
    echo "Using commongen"
    test_data_path=${dev_data_path} # since commongen do not release test set, we use dev set as test set
    source_maxlength=25
    candidate_maxlength=35
    per_device_train_batch_size=4
    per_device_eval_batch_size=32
    gradient_accumulation_steps=16
    using_metrics="bleu,cider"

elif [[ $dataset =~ "wmt18" ]]; then
    echo "Using wmt18"
    source_maxlength=112
    candidate_maxlength=200
    per_device_train_batch_size=4
    per_device_eval_batch_size=16
    gradient_accumulation_steps=16
    using_metrics="bleu"

elif [[ $dataset =~ "xsum" ]]; then
    echo "Using xsum"
    source_maxlength=332
    candidate_maxlength=90
    per_device_train_batch_size=4
    per_device_eval_batch_size=32
    gradient_accumulation_steps=16
    using_metrics="rouge1,rouge2,rougeLsum"

else
    echo "Unknown dataset: ${dataset}"
    echo "Please set the dataset specific parameters in the script"
    exit 1
fi



if [[ $reranker = "PairReranker" ]]; then
    echo "Using PairReranker"
    reranker_type="crosscompare"
    if [ $do_inference = "True" ]; then
        inference_mode="full" # do full for inference for its better performance
        if [ $inference_mode = "full" ]; then
            run_name="test_${dataset}_${reranker}_full_comparison"
        elif [ $inference_mode = "bubble" ]; then
            run_name="test_${dataset}_${reranker}_bubble_comparison"
        fi
        do_train="False"
        do_eval="False"
        do_test="True"
        load_checkpoint="./outputs/${reranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${reranker}/checkpoint-best"
    else
        inference_mode="bubble" # do bubble for inference for its faster speed
        run_name="train_${dataset}_${reranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi

    torchrun \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:${localhost}" \
        --nnodes 1 \
        --nproc_per_node ${n_gpu} \
    train_reranker.py \
        --reranker_type ${reranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model ${candidate_model} \
        --candidate_generation_method ${candidate_generation_method} \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --inference_mode ${inference_mode} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --num_pos 1 \
        --num_neg 1 \
        --loss_type "BCE" \
        --sub_sampling_mode "top_bottom" \
        --reduce_type  "linear" \
        --pooling_type "special" \
        --overwrite_output_dir True \

elif [[ $reranker = "SummaReranker" ]]; then
    echo "Using SummaReranker"
    reranker_type="scr"
    if [ $do_inference = "True" ]; then
        run_name="test_${dataset}_${reranker}"
        do_train="False"
        do_eval="False"
        do_test="True"
        load_checkpoint="./outputs/${reranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${reranker}/checkpoint-best"
    else
        run_name="train_${dataset}_${reranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi
    torchrun \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:${localhost}" \
        --nnodes 1 \
        --nproc_per_node ${n_gpu} \
    train_reranker.py \
        --reranker_type ${reranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model ${candidate_model} \
        --candidate_generation_method ${candidate_generation_method} \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --num_pos 1 \
        --num_neg 1 \
        --loss_type "MoE_BCE" \
        --sub_sampling_mode "top_bottom" \
        --overwrite_output_dir True \

elif [[ $reranker = "SimCLS" ]]; then
    echo "Using SimCLS"
    reranker_type="dual"
    if [ $do_inference = "True" ]; then
        run_name="test_${dataset}_${reranker}"
        do_train="False"
        do_eval="False"
        do_test="True"
        load_checkpoint="./outputs/${reranker_type}/${backbone_name}/train_${checkpoint_trained_dataset}_${reranker}/checkpoint-best"
    else
        run_name="train_${dataset}_${reranker}"
        do_train="True"
        do_eval="True"
        do_test="True"
        load_checkpoint="" # no need to load checkpoint for training
    fi
    torchrun \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:${localhost}" \
        --nnodes 1 \
        --nproc_per_node ${n_gpu} \
    train_reranker.py \
        --reranker_type ${reranker_type} \
        --model_type ${backbone_type} \
        --model_name ${backbone_name} \
        --run_name ${run_name} \
        --train_data_path ${train_data_path} \
        --eval_data_path ${dev_data_path} \
        --test_data_path ${test_data_path} \
        --n_candidates ${n_candidates} \
        --candidate_model ${candidate_model} \
        --candidate_generation_method ${candidate_generation_method} \
        --using_metrics ${using_metrics} \
        --learning_rate ${learning_rate} \
        --source_maxlength ${source_maxlength} \
        --candidate_maxlength ${candidate_maxlength} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --num_train_epochs ${num_train_epochs} \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --do_predict ${do_test} \
        --load_checkpoint "${load_checkpoint}" \
        --max_train_data_size ${max_train_data_size} \
        --max_eval_data_size ${max_eval_data_size} \
        --max_predict_data_size ${max_predict_data_size} \
        --loss_type "simcls" \
        --sub_sampling_mode "none" \
        --overwrite_output_dir True \

else
    echo "Unknown reranker: ${reranker}"
fi
