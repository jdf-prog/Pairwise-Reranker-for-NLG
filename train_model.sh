#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=FID_train
#SBATCH --nodelist=ink-titan
#SBATCH --output ./jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH -n 1
CUDA_LAUNCH_BLOCKING=1 # for debugging
NGPU=1

train_data_path="./data/cnn_dailymail_train_hypo.jsonl"
dev_data_path="./data/cnn_dailymail_val_hypo_min.jsonl"
test_data_path="./data/cnn_dailymail_test_hypo.jsonl"
model_type='dualt5'
model_size="base"
name="bisel_gen_2"
checkpoint_dir="checkpoint/${model_type}-${model_size}"

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
# torchrun --nproc_per_node=$NGPU \
python \
        train_model.py \
        --name "${name}" \
        --train_data ${train_data_path} \
        --eval_data ${dev_data_path} \
        --model_type ${model_type} \
        --model_size ${model_size} \
        --source_maxlength ${text_maxlength} \
        --candidate_maxlength ${text_maxlength} \
        --checkpoint_dir ${checkpoint_dir} \
        --lr 0.00003 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.001 \
        --per_gpu_batch_size 2 \
        --n_candidate 6 \
        --total_step 25000 \
        --warmup_step 3000 \
        --main_port 19008 \
        --use_aux_loss \
        --aux_loss_weight 1 \
