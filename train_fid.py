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
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from functools import partial
warnings.filterwarnings("ignore")
from src.common.utils import (
    str2bool,
    seed_everything
)
from src.dualfid.model_util import (
    build_fid,
    build_tokenizer,
    build_collator,

)
from src.dualfid.data import (
    load_data,
    Dataset
)
from src.common.evaluation import (
    overall_eval
)
from src.dualfid.trainer import (
    FiDTrainer,
)
def main(args):
    seed_everything(args.seed)

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
        train_examples = load_data(args.train_data_path, args)
        train_dataset = Dataset(train_examples, args.n_candidate)
    if args.do_eval:
        eval_examples = load_data(args.eval_data_path, args)
        eval_dataset = Dataset(eval_examples, args.n_candidate)
    if args.do_predict:
        predict_examples = load_data(args.test_data_path, args)
        predict_dataset = Dataset(predict_examples, args.n_candidate)

    # set up data collator
    data_collator_class = build_collator(args.fid_type)
    data_collator = data_collator_class(args.source_maxlength, tokenizer, args.candidate_maxlength)

    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks
    elif args.do_predict:
        args.n_tasks = predict_dataset.n_tasks

    # set up model
    config = {
        "n_tasks": args.n_tasks,
        "use_aux_loss": args.use_aux_loss,
        "top_k_candidates": args.top_k_candidates,
    }

    if args.load_checkpoint:

        model = build_fid(
            args.fid_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config
        )
        state_dict = torch.load(os.path.join(args.load_checkpoint, "pytorch_model.bin"))
        load_result = model.load_state_dict(state_dict)
        if load_result.missing_keys:
            logging.warning(f"Missing keys: {load_result.missing_keys}")
        else:
            logging.info(f"Successfully loaded checkpoint from '{args.load_checkpoint}'")
    else:
        model = build_fid(
            args.fid_type,
            args.model_type,
            args.model_name,
            args.cache_dir,
            config
        )
        logging.info(f"build new model")


    # set up trainer
    training_args = Seq2SeqTrainingArguments(
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
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        log_level=args.log_level,
        report_to=args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        seed=args.seed,
        local_rank=args.local_rank,
        fp16=args.fp16,
        deepspeed=args.deepspeed, #
        sharded_ddp=args.sharded_ddp,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        adafactor=args.adafactor,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        disable_tqdm=False,
        greater_is_better=True,

    )
    logging.info(f"training args: {training_args}")

    compute_metrics = partial(compute_metrics_for_seq2seq_trainer, tokenizer=tokenizer)
    trainer = FiDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        # set up wandb
        if args.report_to == "wandb":
            wandb.init(project="Fusion in Decoder", group=args.fid_type, name=args.run_name)
            wandb.config.update(args)

        if args.evaluate_first_step:
            metrics = trainer.evaluate()
            logging.info(f"Evaluate first step: \n{metrics}")

        logging.info("Start training")
        outputs = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        logging.info("Training finished")
        global_step, training_loss = outputs.global_step, outputs.training_loss
        metrics = outputs.metrics
        logging.info(f"global_step = {global_step}, average loss = {training_loss}")
        for key, value in metrics.items():
            logging.info(f"{key} = {value}")

        logging.info("Saving model")
        best_checkpoint_folder = os.path.join(args.output_dir, "checkpoint-best")
        trainer.save_model(best_checkpoint_folder)
        torch.save(model.args, os.path.join(best_checkpoint_folder, "config.bin"))

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
            with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
                json.dump(predictions.tolist(), f)
            with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
                json.dump(labels.tolist(), f)
            logging.info(f"predictions saved to {output_path}")


def compute_metrics_for_seq2seq_trainer(eval_pred: EvalPrediction, tokenizer) -> Dict[str, float]:
    """
    Compute metrics for seq2seq trainer
    Note that when passing to the seq2seq trainer,
    you should use partial function to pass the tokenizer at first
    """
    generated_ids, label_ids = eval_pred
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    label_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    metrics = overall_eval(generated_text, label_text, metrics=["rouge1", "rouge2", "rougeL", "bleu", "rougeLsum"])
    metrics["dev_score"] = metrics["rouge2"] # dev score used for save checkpoint
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # model config
    parser.add_argument("--fid_type", type=str, choices=[
        "fid", "dualfid"
    ], default="t5")
    parser.add_argument("--model_type", type=str, choices=["t5", "bart"], default="t5")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_aux_loss", type=str2bool, default=False)
    parser.add_argument("--top_k_candidates", type=int, default=-1)

    # data config
    parser.add_argument("--n_candidate", type=int, default=-1)
    parser.add_argument("--candidate_generation_method", type=str, default=None)
    parser.add_argument("--candidate_model", type=str, default=None)
    parser.add_argument("--source_maxlength", type=int, default=512)
    parser.add_argument("--candidate_maxlength", type=int, default=128)

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
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=10e10)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=0) # Overrides any effect of :obj:`warmup_ratio`.
    parser.add_argument("--lr_scheduler_type", type=str, choices=[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ], default="linear")
    parser.add_argument('--adafactor', type=bool, default=True)

    # evaluation hyperparameters
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--evaluate_first_step", type=str2bool, default=False)
    parser.add_argument("--evaluation_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=0)

    # predict config
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--save_predictions", type=str2bool, default=True)

    # generation config
    parser.add_argument("--generation_max_length", type=int, default=128)
    parser.add_argument("--generation_num_beams", type=int, default=2)

    # logging
    parser.add_argument("--logging_first_step", type=str2bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--log_level", type=str, default="passive",
        choices=["passive", "info", "debug", "warning", "error", "critical"])
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="basic") # wandb run name


    # save config
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", type=str2bool, default=False)
    parser.add_argument("--save_strategy", type=str, choices=[
        "steps", "epoch", "no"
    ], default="epoch")
    parser.add_argument("--save_steps", type=int, default=0)

    # metrics config
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default="dev_score")

    # init args
    args = parser.parse_args()
    args.load_best_model_at_end = args.do_train and args.do_predict
    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.fid_type}/{args.model_name}/{args.run_name}"

    args.cache_dir = "./hf_models/" + args.model_name.split('/')[-1] + "/"

    args.candidate_generation_methods = args.candidate_generation_method.split('+')
    args.candidate_models = args.candidate_model.split('+')

    # set up logging
    if args.log_level == "passive":
        args.log_level = "info"
    logging.basicConfig(level="INFO")
    logging.info("args: %s", args)
    main(args)



