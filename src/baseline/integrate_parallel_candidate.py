"""
    This file integrates the summary candidates generated using scripts:
        generate_parallel_candidates.sh
"""

import argparse
import sys
import os
import torch
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (
    load_pkl_candidates,
    load_pkl_sources_and_targets,
    save_pkl_candidates,
    save_pkl_sources_and_targets,
    get_parallel_candidate_types,
)
from common.utils import (
    seed_everything,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--dataset', type=str, default = "cnndm",
        choices= ["cnndm", "xsum", "reddit", 'wmt18', 'commongen'])
    parser.add_argument('--set', type=str, default = "test",
        choices= ["test", "val", "train"])

    args = parser.parse_args()

    seed_everything(args.seed)

    parallel_candidate_types = get_parallel_candidate_types(args.dataset, args.set)
    shard_idxs = {}
    for parallel_candidate_type in parallel_candidate_types:
        model, generation_method, start_idx, end_idx = parallel_candidate_type
        if model not in shard_idxs:
            shard_idxs[model] = {}
        if generation_method not in shard_idxs[model]:
            shard_idxs[model][generation_method] = {}
        if "shard_idxs" not in shard_idxs[model][generation_method]:
            shard_idxs[model][generation_method]["shard_idxs"] = {}
        assert start_idx not in shard_idxs[model][generation_method]["shard_idxs"], f"start_idx {start_idx} already exists"
        shard_idxs[model][generation_method]["shard_idxs"][start_idx] = end_idx

    for model in shard_idxs:
        for generation_method in shard_idxs[model]:
            print("---"*10)
            print(f"Integrating model: {model}\ngeneration_method: {generation_method}")
            integrated_candidates = []
            integrated_sources = []
            integrated_targets = []
            shard_idxs = shard_idxs[model][generation_method]["shard_idxs"]
            start_idxs = sorted(list(shard_idxs.keys()))
            end_idxs = [shard_idxs[start_idx] for start_idx in start_idxs]
            assert start_idxs[1:] == end_idxs[:-1], "start_idxs and end_idxs are not aligned"

            print(f"start_idxs: {start_idxs}")
            print(f"end_idxs: {end_idxs}")
            for start_idx, end_idx in zip(start_idxs, end_idxs):
                assert start_idx < end_idx, f"start_idx {start_idx} is not less than end_idx {end_idx}"
                candidates = load_pkl_candidates(args.dataset, args.set, generation_method, model, start_idx, end_idx)
                sources, targets = load_pkl_sources_and_targets(args.dataset, args.set, start_idx, end_idx)
                integrated_candidates.extend(candidates)
                integrated_sources.extend(sources)
                integrated_targets.extend(targets)
            assert len(integrated_candidates) == end_idx, f"len(integrated_candidates) {len(integrated_candidates)} is not equal to end_idx {end_idx}"
            assert len(integrated_sources) == end_idx, f"len(integrated_sources) {len(integrated_sources)} is not equal to end_idx {end_idx}"
            assert len(integrated_targets) == end_idx, f"len(integrated_targets) {len(integrated_targets)} is not equal to end_idx {end_idx}"
            print("length of integrated: ", len(integrated_candidates))
            print("Saving integrated sources, targets and candidates")
            # save_pkl_candidates(args.dataset, args.set, generation_method, model, integrated_candidates)
            print(f"Saved {generation_method} {model} {args.set} candidates")
            # save_pkl_sources_and_targets(args.dataset, args.set, integrated_sources, integrated_targets)
            print(f"Saved {generation_method} {model} {args.set} sources and targets")









