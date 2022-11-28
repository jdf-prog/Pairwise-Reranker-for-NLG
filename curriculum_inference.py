# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import torch
import transformers
import numpy as np
import os
import warnings
import logging
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.cuda.amp import autocast
warnings.filterwarnings("ignore")
from src.common.utils import (
    str2bool,
    empty2zero,
    seed_everything
)
from src.dualfid.model_util import (
    build_reranker,
    build_tokenizer,
    build_collator
)
from src.dualfid.data import (
    load_data,
    Dataset
)
from src.dualfid.curriculum import (
    CurriculumDataset,
)

def main(args):
    seed_everything(args.seed)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")

    # set up tokenizer
    tokenizer = build_tokenizer(args.model_type, args.model_name, args.cache_dir)
    # set up data collator, also add prefix as new tokens to tokenizer
    data_collator = build_collator(
        args.reranker_type, tokenizer,
        args.source_maxlength, args.candidate_maxlength,
        curriculum_learning=True
    )
    # set up dataset
    examples = load_data(args.data_path, args, max_size=args.max_data_size)
    if args.num_shards > 1 and args.shard_id >= 0:
        examples = np.array_split(examples, args.num_shards)
        offset = sum([len(examples[i]) for i in range(args.shard_id)])
        examples = examples[args.shard_id]
        logging.info(f"shard_id: {args.shard_id}, offset: {offset}")
    else:
        offset = 0
    dataset = CurriculumDataset(examples, args.n_candidates)
    args.n_tasks = dataset.n_tasks

    # set up model
    config = {
        "n_tasks": args.n_tasks,
        "num_pos": args.num_pos,
        "num_neg": args.num_neg,
        "sub_sampling_ratio": args.sub_sampling_ratio,
        "sub_sampling_mode": args.sub_sampling_mode,
        "loss_type": args.loss_type,
        "new_num_tokens": len(tokenizer),
        "training_data_size": len(dataset) if dataset else 0,
        "n_candidates": args.n_candidates,
    }
    if args.load_checkpoint:
        # config = torch.load(os.path.join(args.load_checkpoint, "config.bin"))
        model = build_reranker(
            args.reranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
        state_dict = torch.load(os.path.join(args.load_checkpoint, "pytorch_model.bin"), map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            logging.warning(f"Missing keys: {load_result.missing_keys}")
        else:
            logging.info(f"Successfully loaded checkpoint from '{args.load_checkpoint}'")
    else:
        model = build_reranker(
            args.reranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config,
            tokenizer,
        )
        logging.info(f"build new model")
    model = model.to(device)
    model.eval()

    # curriculum inference
    preds = []
    dif_scores = []
    losses = []
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    with tqdm(total=len(dataloader), desc="Do curriculum inference") as progress_bar:
        for inputs in dataloader:
            assert inputs['scores'].shape[2] == 2
            with torch.no_grad():
                if args.fp16:
                    with autocast():
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                pred = outputs['preds'].detach().cpu().numpy()
                loss = outputs['loss'].detach().cpu().numpy()
                dif_score = inputs["scores"][:, :, 0] - inputs["scores"][:, :, 1]
                dif_score = dif_score.detach().cpu().numpy()
                preds.append(pred)
                losses.append(loss)
                dif_scores.append(dif_score.sum(-1))
                progress_bar.update(1)
    preds = np.concatenate(preds)
    dif_scores = np.concatenate(dif_scores)
    dataset_indices = dataset.index_to_data
    for i in range(len(dataset_indices)):
        dataset_indices[i][0] += offset
    dir_path = args.output_dir
    # save indices
    shard_postfix = f"_shard_{args.shard_id}" if args.num_shards > 1 and args.shard_id >= 0 else ""
    np.save(dir_path / f"preds{shard_postfix}.npy", preds)
    np.save(dir_path / f"dif_scores{shard_postfix}.npy", dif_scores)
    np.save(dir_path / f"dataset_indices{shard_postfix}.npy", dataset_indices)

    logging.info(f"acc: {(preds * dif_scores > 0).mean()}")
    if args.num_shards > 1 and args.shard_id >= 0:
        logging.info("Done curriculum inference for shard %d", args.shard_id)
    else:
        logging.info("Done curriculum inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # model config
    parser.add_argument("--reranker_type", type=str, choices=[
        "scr", "dual", "crosscompare", "dualcompare"
    ], default="sc")
    parser.add_argument("--model_type", type=str, choices=[
        "roberta", "bert", "t5", 'deberta', 'xlm-roberta'
    ], default="roberta")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--loss_type", type=str, choices=[
      "BCE", "infoNCE", "ListNet", "ListMLE", "p_ListMLE",
      "triplet", "triplet_v2", "triplet_simcls", "MoE_BCE", "MSE", "ApproxNDCG",
      "ranknet", "MoE_ranknet", "lambdarank"
    ], default="BCE")

    # data config
    parser.add_argument("--n_candidates", type=int, default=-1)
    parser.add_argument("--candidate_generation_method", type=str, default=None)
    parser.add_argument("--candidate_model", type=str, default=None)
    parser.add_argument("--source_maxlength", type=int, default=128)
    parser.add_argument("--candidate_maxlength", type=int, default=512)
    parser.add_argument("--num_pos", type=int, default=0)
    parser.add_argument("--num_neg", type=int, default=0)
    parser.add_argument("--sub_sampling_ratio", type=float, default=0.4)
    parser.add_argument("--sub_sampling_mode", type=str, choices=[
        "uniform", "top_bottom", "top_random", "random_bottom",
        "importance", "random", "poisson_dynamic"
    ], default="top_bottom")
    parser.add_argument("--max_data_size", type=int, default=-1)
    parser.add_argument("--using_metrics", type=str, default="rouge1+rouge2+rougeLsum")

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument("--shard_id", type=empty2zero, default=-1)
    parser.add_argument("--num_shards", type=empty2zero, default=-1)

    # logging
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])

    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)

    # init args
    args = parser.parse_args()
    # set up default output dir
    args.output_dir = Path(args.data_path).parent if args.output_dir is None else Path(args.output_dir)
    args.cache_dir = "./hf_models/" + args.model_name.split('/')[-1] + "/"
    args.candidate_generation_methods = args.candidate_generation_method.split('+')
    args.candidate_models = args.candidate_model.split('+')
    args.metrics = args.using_metrics.split('+')
    # set up logging
    if args.log_level == "passive":
        args.log_level = "info"
    logging.basicConfig(level="INFO")
    logging.info("args: %s", args)
    main(args)
