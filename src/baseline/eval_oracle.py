"""
    Eval results will be continuously saved to ../../data/prepared/{dataset_name}/{set_name}/dataset.jsonl
"""
import argparse
import sys
import os
import psutil
from tqdm.contrib.concurrent import process_map
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (
    load_pkl_candidates,
    save_pkl_cand_scores,
    get_candidate_types,
    get_candidate_metrics,
    load_pkl_sources_and_targets
)
from common.dataset import (
    CustomDataset,
)
from common.utils import (
    seed_everything,
    str2bool,
)
from common.evaluation import (
    overall_eval
)
from pathlib import Path


def main(args):
    # seed
    seed_everything(args.seed)

    # prepare metrics
    metrics = []
    if args.eval_rouge:
        metrics.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    if args.eval_bleu:
        metrics.extend(["bleu"])
    if args.eval_bleurt:
        metrics.extend(["bleurt"])

    # model and generation_method of current computed candidates
    types = get_candidate_types(args.dataset, args.set)
    sources, targets = load_pkl_sources_and_targets(args.dataset, args.set)
    # add candidates if the curent model and generation_method are not in the dataset
    for model_name, generation_method in types:
        print("Checking candidates scores from -- model:{} \t generation method:{}".format(model_name, generation_method))
        scored_metrics = get_candidate_metrics(args.dataset, args.set, model_name, generation_method)
        print("Scored metrics: {}".format(scored_metrics))
        to_score_metrics = [metric for metric in metrics if metric not in scored_metrics]
        if len(to_score_metrics) == 0:
            print("All metrics are already computed for -- model:{} \t generation method:{}".format(model_name, generation_method))
            continue
        print("Computing metrics: {}".format(to_score_metrics))
        candidates = load_pkl_candidates(args.dataset, args.set, generation_method, model_name)
        scores = overall_eval(candidates, targets, to_score_metrics, args.num_workers)
        for k, v in scores.items():
            save_pkl_cand_scores(args.dataset, args.set, generation_method, model_name, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--set", type=str, default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    # metrics
    parser.add_argument('--eval_rouge', type = str2bool, default = True)
    parser.add_argument('--eval_bleu', type = str2bool, default = True)
    parser.add_argument('--eval_bleurt', type = str2bool, default = False)
    args = parser.parse_args()
    print(args)
    main(args)
