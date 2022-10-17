# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import os
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchmetrics.text.rouge import ROUGEScore
import wandb
import os
import pprint
import src.dualfid.slurm
import src.dualfid.util
import src.dualfid.data
import src.dualfid.model
import warnings
from src.dualfid.options import Options
warnings.filterwarnings("ignore")

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_score, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()


    if opt.is_main:
        wandb.init(project="DualFID", group=opt.model_type + '-' + opt.model_size, name=opt.name)
        wandb.config.update(opt)
        wandb_log = {}

    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(train_dataloader):
            step += 1

            (index, target_ids, target_mask, context_ids, context_mask, scores) = batch
            # compute the generation loss
            generation_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=target_ids.cuda()
            )[0]
            # compute the mutli-task auxiliary loss, for this batch
            if opt.use_aux_loss:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    _, aux_loss = model.module.compute_auxiliary_loss(scores)
                else:
                    _, aux_loss = model.compute_auxiliary_loss(scores)
                train_loss = generation_loss + aux_loss * opt.aux_loss_weight
            else:
                aux_loss = 0
                train_loss = generation_loss

            train_loss /= opt.accumulation_steps
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.dualfid.util.average_main(train_loss, opt)
            curr_loss += train_loss

            if step % opt.eval_freq == 0:
                eval_result = evaluate(model, eval_dataset, tokenizer, collator, opt)
                wandb_log.update(eval_result)
                dev_score = eval_result['dev_score']
                model.train()
                if opt.is_main:
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        src.dualfid.util.save(model, optimizer, scheduler, step, best_dev_score,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {dev_score:.2f}ROUGE2 |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_score, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            # log on the wandb
            if opt.is_main:
                wandb_log.update({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'generation_loss': generation_loss,
                    'aux_loss': aux_loss,
                })
                if opt.use_dual_encoder:
                    wandb_log.update({
                        "source_encoder_lr": scheduler.get_lr()[0],
                        "target_encoder_lr": scheduler.get_lr()[1],
                        "decoder_lr": scheduler.get_lr()[2],
                    })
                else:
                    wandb_log.update({
                        "lr": scheduler.get_last_lr()[0],
                    })
                wandb.log(wandb_log, step=step)

            if opt.is_main and step % opt.save_freq == 0:
                src.dualfid.util.save(model, optimizer, scheduler, step, best_dev_score,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    rouge_score = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )
    model.eval()
    rouge_scores_sel = []
    rouge_scores_gen = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_masks, scores) = batch
            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_masks.cuda(),
                max_length=opt.max_length,
                num_beams=opt.num_beams,
                min_length=opt.min_length,
            )
            if opt.use_aux_loss:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    preds, aux_loss = model.module.compute_auxiliary_loss(scores)
                else:
                    preds, aux_loss = model.compute_auxiliary_loss(scores)
                for k, pred in enumerate(preds):
                    select_idx = torch.argmax(pred)
                    score = scores[k][select_idx]
                    rouge_scores_sel.append(score)
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['target']
                score = rouge_score(ans, gold)
                rouge_scores_gen.append(score)
    result = {
        "gen": {
            'rouge1': np.mean([r['rouge1_fmeasure'] for r in rouge_scores_gen]),
            'rouge2': np.mean([r['rouge2_fmeasure'] for r in rouge_scores_gen]),
            'rougeL': np.mean([r['rougeL_fmeasure'] for r in rouge_scores_gen]),
        },
        "dev_score": np.mean([r['rouge2_fmeasure'] for r in rouge_scores_gen]),
    }
    if opt.use_aux_loss:
        result.update({
            "sel":{
                'rouge1': np.mean([r['rouge1'] for r in rouge_scores_sel]),
                'rouge2': np.mean([r['rouge2'] for r in rouge_scores_sel]),
                'rougeL': np.mean([r['rougeL'] for r in rouge_scores_sel]),
            }})

    return result

if __name__ == "__main__":
    options = Options()
    options.add_train_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.dualfid.slurm.init_distributed_mode(opt)
    src.dualfid.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = (checkpoint_path / 'checkpoint' / 'latest').exists()
    if opt.is_distributed:
        print(f"rank {opt.global_rank} is waiting for barrier")
        torch.distributed.barrier()
    print(f"rank {opt.global_rank} barrier passed")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.dualfid.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.dualfid.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
        n_tasks=opt.n_tasks,
    )
    train_dataset = src.dualfid.data.Dataset(train_examples, opt.n_candidate)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.dualfid.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
        n_tasks=opt.n_tasks,
    )
    eval_dataset = src.dualfid.data.Dataset(eval_examples, opt.n_candidate)
    assert train_dataset.n_tasks == eval_dataset.n_tasks
    opt.n_tasks = train_dataset.n_tasks

    opt.use_dual_encoder = "dual" in opt.model_type
    model_name = opt.model_type + '-' + opt.model_size
    if opt.model_type == 't5':
        model_name = "t5-" + opt.model_size
        model_class = src.dualfid.model.FiDT5
        hf_model_class = transformers.T5ForConditionalGeneration
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        collator = src.dualfid.data.FiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == "dualt5":
        model_name = "t5-" + opt.model_size
        model_class = src.dualfid.model.DualFiDT5
        hf_model_class = transformers.T5ForConditionalGeneration
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
        collator = src.dualfid.data.DualFiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == 'bart':
        model_name = "facebook/bart-large-cnn"
        model_class = src.dualfid.model.FiDBART
        hf_model_class = transformers.BartForConditionalGeneration
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
        collator = src.dualfid.data.FiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == "dualbart":
        model_name = "facebook/bart-large-cnn"
        model_class = src.dualfid.model.DualFiDBART
        hf_model_class = transformers.BartForConditionalGeneration
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name)
        collator = src.dualfid.data.DualFiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    else:
        raise NotImplementedError



    if not checkpoint_exists and opt.model_path == "none":
        hf_model = hf_model_class.from_pretrained(model_name)
        hf_model.config.n_tasks = opt.n_tasks
        hf_model.config.use_aux_loss = opt.use_aux_loss
        hf_model.config.top_k_candidates = opt.top_k_candidates
        model = model_class(hf_model.config)
        model.load_hfm(hf_model.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.dualfid.util.set_optim(opt, model)
        step, best_dev_score = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_score = \
            src.dualfid.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_score = \
            src.dualfid.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")


    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_score,
        checkpoint_path
    )
