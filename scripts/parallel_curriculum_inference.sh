num_shards=10

for shard_id in $(seq 0 $((num_shards-1))); do
    sbatch curriculum_inference.sh \
        "$num_shards" \
        "$shard_id"
done
