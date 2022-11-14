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
    seed_everything
)
from src.dualfid.trainer import (
    compute_metrics_for_crosscompare,
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
from src.dualfid.trainer import (
    RerankerTrainer,
)
from src.dualfid.curriculum import (
    CurriculumDataset,
    CurriculumCallback,
    compute_metrics_for_curriculum,
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
        # source_prefix="source text: ",
        # candidate1_prefix="candidate text: ",
        # candidate2_prefix="candidate text: ",
        curriculum_learning=args.curriculum_learning
    )

    # set up dataset
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if args.do_train:
        train_examples = load_data(args.train_data_path, args, mode='train')
        train_dataset = Dataset(train_examples, args.n_candidates)
    if args.do_eval:
        eval_examples = load_data(args.eval_data_path, args, mode='val')
        eval_dataset = Dataset(eval_examples, args.n_candidates)
    if args.do_predict:
        predict_examples = load_data(args.test_data_path, args, mode='predict')
        predict_dataset = Dataset(predict_examples, args.n_candidates)



    if args.do_train:
        if args.do_eval:
            assert train_dataset.n_tasks == eval_dataset.n_tasks
        args.n_tasks = train_dataset.n_tasks
    elif args.do_predict:
        args.n_tasks = predict_dataset.n_tasks

    # set up model
    config = {
        "n_tasks": args.n_tasks,
        "num_pos": args.num_pos,
        "num_neg": args.num_neg,
        "sub_sampling_ratio": args.sub_sampling_ratio,
        "sub_sampling_mode": args.sub_sampling_mode,
        "loss_type": args.loss_type,
        "new_num_tokens": len(tokenizer),
        "training_data_size": len(train_dataset) if train_dataset else 0,
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
        state_dict = torch.load(os.path.join(args.load_checkpoint, "pytorch_model.bin"))
        load_result = model.load_state_dict(state_dict)
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
        label_names=args.label_names,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        adafactor=args.adafactor,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        disable_tqdm=False,
        greater_is_better=True,
    )
    # custom argument for curriculum learning
    training_args.curriculum_learning = args.curriculum_learning
    training_args.num_curriculum = args.num_curriculum
    training_args.curriculum_size = args.curriculum_size

    logging.info(f"training args: {training_args}")
    logging.info(f"model config: {config}")

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CurriculumCallback]
    )

    if args.do_train:
        # set up wandb
        if args.report_to == "wandb":
            wandb.init(project="Reranker", group=args.reranker_type, name=args.run_name)
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
            with open(os.path.join(args.output_dir, "predictions.pt"), "wb") as f:
                torch.save(predictions, f)
            with open(os.path.join(args.output_dir, "labels.pt"), "wb") as f:
                torch.save(labels, f)
            logging.info(f"predictions saved to {output_path}")



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
      "triplet", "triplet_v2", "triplet_simcls", "MoE_BCE", "MSE"
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
    parser.add_argument("--max_train_data_size", type=int, default=25000)
    parser.add_argument("--max_eval_data_size", type=int, default=-1)
    parser.add_argument("--max_predict_data_size", type=int, default=-1)

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
    parser.add_argument("--save_total_limit", type=int, default=10)

    # metrics config
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default="dev_score")

    # curriculum learning
    parser.add_argument("--curriculum_learning", type=str2bool, default=False)
    parser.add_argument("--num_curriculum", type=int, default=1)
    parser.add_argument("--curriculum_size", type=int, default=1000)

    # init args
    args = parser.parse_args()
    args.load_best_model_at_end = args.do_train and args.do_predict
    # set up default output dir
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.reranker_type}/{args.model_name}/{args.run_name}"
    args.cache_dir = "./hf_models/" + args.model_name.split('/')[-1] + "/"
    args.label_names = ["scores"]
    args.candidate_generation_methods = args.candidate_generation_method.split('+')
    args.candidate_models = args.candidate_model.split('+')
    args.local_rank = os.environ.get("LOCAL_RANK", args.local_rank)

    # prepare for curriculum learning
    if args.curriculum_learning:
        print("Using curriculum learning")
        Dataset = CurriculumDataset
        compute_metrics = compute_metrics_for_curriculum
        args.num_train_epochs *= args.num_curriculum # multiply epochs
    else:
        compute_metrics = compute_metrics_for_crosscompare


    # set up logging
    if args.log_level == "passive":
        args.log_level = "info"
    logging.basicConfig(level="INFO")
    logging.info("args: %s", args)
    main(args)
