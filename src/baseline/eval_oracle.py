"""
    Eval results will be continuously saved to ../../data/prepared/{dataset_name}/{set_name}/dataset.jsonl
"""
import argparse
import sys
import os
import psutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data import (
    load_pkl_candidates,
    load_raw_dataset,
    save_pkl_cand_scores,
    save_pkl_sources_and_targets,
    get_candidate_types,
    get_candidate_metrics,
    load_pkl_sources_and_targets,
    load_prepared_dataset,
    save_prepared_dataset
)
from common.dataset import (
    CustomDataset,
)
from common.utils import (
    seed_everything,
    str2bool,
)
from common.evaluation import (
    overall_eval,
    SUPPORTED_METRICS
)
from pathlib import Path


def main(args):
    # seed
    seed_everything(args.seed)

    # prepare metrics
    if 'rouge' in args.metrics:
        args.metrics.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        args.metrics.remove('rouge')
    metrics = args.metrics
    assert set(metrics).issubset(set(SUPPORTED_METRICS)), \
        "Unsupported metrics: {}".format(set(SUPPORTED_METRICS)-set(metrics))


    # model and generation_method of current computed candidates
    types = get_candidate_types(args.dataset, args.set)
    try:
        print("load pkl sources and targets")
        sources, targets = load_pkl_sources_and_targets(args.dataset, args.set)
    except:
        print("pkl sources and targets not found, load from raw dataset")
        ids, sources, targets = load_raw_dataset(args.dataset, args.set, load_shuffle=args.load_shuffle)
        save_pkl_sources_and_targets(args.dataset, args.set, sources, targets)

    # add candidates if the curent model and generation_method are not in the dataset
    for model_name, generation_method in types:
        print("Checking candidates scores from -- model:{} \t generation method:{}".format(model_name, generation_method))
        scored_metrics = get_candidate_metrics(args.dataset, args.set, model_name, generation_method)
        print("Scored metrics: {}".format(scored_metrics))
        if args.overwrite:
            to_score_metrics = metrics
            print("Overwrite mode: all metrics will be scored")
        else:
            to_score_metrics = [metric for metric in metrics if metric not in scored_metrics]
            if len(to_score_metrics) == 0:
                print("All metrics are already computed for -- model:{} \t generation method:{}".format(model_name, generation_method))
                continue
        print("Computing metrics: {}".format(to_score_metrics))
        candidates = load_pkl_candidates(args.dataset, args.set, generation_method, model_name)
        scores = overall_eval(candidates, targets, to_score_metrics, args.num_workers)
        if args.save_scores:
            for k, v in scores.items():
                save_pkl_cand_scores(args.dataset, args.set, generation_method, model_name, k, v)

    ds = load_prepared_dataset(args.dataset, args.set, metrics=metrics)
    ds.analyze_oracle()
    if args.save_prepared:
        save_prepared_dataset(args.dataset, args.set, ds)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnndm")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--load_shuffle", type=str2bool, default=False)
    # metrics
    parser.add_argument("--metrics", type=str, default="rouge,bleu",
        help="metrics to compute, support rouge, bleu, bleurt, cider, spice, bleu4")
    parser.add_argument("--save_scores", type=str2bool, default=True, help="save scores to disk")
    parser.add_argument("--save_prepared", type=str2bool, default=False, help="save prepared dataset for training")
    args = parser.parse_args()
    args.metrics = args.metrics.split(",")
    print(args)
    main(args)
