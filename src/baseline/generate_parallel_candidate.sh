# dataset='wmt18'
# model_type='opus-mt'
# model='Helsinki-NLP/opus-mt-zh-en'
# model_name='opus_mt'
# load_model=False
# load_model_path=""
# set='train'
# partition=""
# inference_bs=15
# generation_method="beam_search"

# num_shards=1
# shard_size=50

dataset=wmt18
model_type="opus-mt"
model="Helsinki-NLP/opus-mt-zh-en"
model_name="opus_mt"
load_model=False
load_model_path=""
set='train'
partition='full'
inference_bs=10
generation_method="top_p_sampling"

num_shards=5
shard_size=100000

for shard_id in $(seq 0 $((num_shards - 1)))
do
    sbatch _generate_parallel_candidate.sh \
        "$dataset" \
        "$model_type" \
        "$model" \
        "$model_name" \
        "$load_model" \
        "$load_model_path" \
        "$set" \
        "$partition" \
        "$inference_bs" \
        "$generation_method" \
        "$num_shards" \
        "$shard_size" \
        "$shard_id"
done
