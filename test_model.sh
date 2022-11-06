#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=bash
#SBATCH --nodelist=ink-gary
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1

train_data_path="./data/cnn_dailymail_train_hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"
model_type='dualt5'
model_size="base"
name="sel_gen_BCE"

checkpoint_dir="checkpoint/${model_type}-${model_size}"
model_path="${checkpoint_dir}/${name}/checkpoint/best_dev"
if [ ${model_type} == 't5' ]; then
        text_maxlength=512
elif [ ${model_type} == 'bart' ]; then
        text_maxlength=1024
elif [ ${model_type} == 'dualt5' ]; then
        text_maxlength=512
elif [ ${model_type} == 'dualbart' ]; then
        text_maxlength=1024

else
        echo "model type not supported"
        exit 1
fi


echo "model type: ${model_type}"
echo "model size: ${model_size}"
echo "name: ${name}"
echo "text_maxlength: ${text_maxlength}"

nvidia-smi
python test_model.py \
        --model_type ${model_type} \
        --model_size ${model_size} \
        --name ${name} \
        --checkpoint_dir ${checkpoint_dir} \
        --model_path ${model_path} \
        --eval_data ${test_data_path} \
        --source_maxlength ${text_maxlength} \
        --candidate_maxlength ${text_maxlength} \
        --per_gpu_batch_size 2 \
        --n_candidates 6 \
        --main_port 19001 \
