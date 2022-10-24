# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import sys
import json
import argparse
import torch
import transformers
import numpy as np
import wandb
import os
import pprint
import warnings
import logging
from transformers import TrainingArguments
from transformers.trainer_utils import PredictionOutput
warnings.filterwarnings("ignore")
from src.common.utils import (
    str2bool,
)
from src.dualfid.trainer import (
    compute_metrics,
)
from src.dualfid.model_util import (
    build_reranker,
    build_reranker_from_checkpoint,
    save_reranker_checkpoint,
    build_tokenizer
)
from src.dualfid.data import (
    get_data_collator_class,
    load_data,
    Dataset
)
from src.dualfid.trainer import (
    RerankerTrainer,
)

def main(args):
    # set up logging
    if args.log_level == "passive":
        args.log_level = "info"
    logging.basicConfig(level=args.log_level.upper())

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info(f"device: {device}, n_gpu: {n_gpu}")

    # set up tokenizer
    tokenizer = build_tokenizer(args.model_type, args.model_name, args.cache_dir)

    # set up dataset
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if args.do_train:
        train_examples = load_data(args.train_data_path)
        train_dataset = Dataset(train_examples, args.n_candidate)
    if args.do_eval:
        eval_examples = load_data(args.eval_data_path)
        eval_dataset = Dataset(eval_examples, args.n_candidate)
    if args.do_predict:
        predict_examples = load_data(args.test_data_path)
        predict_dataset = Dataset(predict_examples, args.n_candidate)

    # set up data collator
    data_collator_class = get_data_collator_class(args.reranker_type)
    data_collator = data_collator_class(args.source_maxlength, tokenizer, args.candidate_maxlength)

    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks

    # set up model
    if args.load_checkpoint:
        model, optimizer, scheduler, config = build_reranker_from_checkpoint(
            args.reranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            args.load_checkpoint
        )
    else:
        assert args.do_train, "Must train a new model if not loading from checkpoint"
        optimizer, scheduler = None, None
        config = {
            "n_tasks": args.n_tasks,
            "num_pos": args.num_pos,
            "num_neg": args.num_neg,
            "loss_type": args.loss_type,
        }
        model = build_reranker(
            args.reranker_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config
        )

    # set up trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        save_steps=args.save_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        disable_tqdm=False,
        greater_is_better=True,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed, #
        sharded_ddp=args.sharded_ddp,
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy
    )
    logging.info(f"training args: {training_args}")

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    # set up wandb
    if args.report_to == "wandb":
        wandb.init(project="Reranker", group=args.reranker_type, name=args.run_name)
        wandb.config.update(args)

    if args.do_train:
        logging.info("Start training")
        outputs = trainer.train()
        logging.info("Training finished")
        global_step, training_loss = outputs.global_step, outputs.training_loss
        metrics = outputs.metrics
        logging.info(f"global_step = {global_step}, average loss = {training_loss}")
        for key, value in metrics.items():
            logging.info(f"{key} = {value}")

        logging.info("Saving model")
        best_checkpoint_folder = "checkpoint-best"
        save_reranker_checkpoint(
            trainer,
            model,
            args.output_dir,
            best_checkpoint_folder
        )

    if args.do_predict:
        logging.info("Start predicting")
        outputs: PredictionOutput = trainer.predict(predict_dataset)
        predictions = outputs.predictions
        labels = outputs.label_ids
        metrics = outputs.metrics
        logging.info(f"metrics: {metrics}")

        # save predictions
        if args.save_predictions:
            if args.output_dir is None:
                raise ValueError("output_dir must be set to save predictions")
            output_path = os.path.join(args.output_dir, "predictions.json")
            with open(output_path, "w") as f:
                json.dump(predictions.tolist(), f)
            logging.info(f"predictions saved to {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--reranker_type", type=str, choices=[
        "scr", "dual", "listwise", "pairwise", "pairwise_gen"
    ], default="sc")
    parser.add_argument("--model_type", type=str, choices=[
        "roberta", "bert", "t5"
    ], default="roberta")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--loss_type", type=str, choices=[
      "BCE", "infoNCE", "ListNet", "ListMLE", ""
    ], default="BCE")

    # data config
    parser.add_argument("--n_candidate", type=int, default=-1)
    parser.add_argument("--source_maxlength", type=int, default=128)
    parser.add_argument("--candidate_maxlength", type=int, default=512)
    parser.add_argument("--num_pos", type=int, default=1)
    parser.add_argument("--num_neg", type=int, default=1)

    # running config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--fp16', type=str2bool, default=True)
    parser.add_argument('--deepspeed', type=str, default=None) # "ds_config.json"
    parser.add_argument('--sharded_ddp', type=str, default="simple")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # mode
    parser.add_argument("--do_train", type=str2bool, default=True)
    parser.add_argument("--do_eval", type=str2bool, default=True)
    parser.add_argument("--do_predict", type=str2bool, default=True)

    # training hyperparameters
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1000) # Overrides any effect of :obj:`warmup_ratio`.
    parser.add_argument("--lr_scheduler_type", type=str, choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], default="linear")

    # evaluation hyperparameters
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=500)


    # predict config
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    # logging
    parser.add_argument("--logging_first_step", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="basic") # wandb run name

    # save config
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)

    # metrics config
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="dev_score")

    # init args
    args = parser.parse_args()
    args.load_best_model_at_end = args.do_train and args.do_predict
    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.reranker_type}/{args.model_name}/{args.run_name}"

    args.cache_dir = "./hf_models/" + args.model_name.split('/')[-1] + "/"
    args.label_names = ["scores"]
    args.evaluation_strategy = "steps"

    main(args)
