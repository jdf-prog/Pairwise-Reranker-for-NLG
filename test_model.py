# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os
import torch
import transformers
import numpy as np
import torch.distributed as dist
import warnings
import src.dualfid.slurm
import src.dualfid.util
import src.dualfid.data
import src.dualfid.model
from torch.utils.data import DataLoader, SequentialSampler
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
from pathlib import Path
from src.dualfid.options import Options
warnings.filterwarnings("ignore")


def evaluate(model, dataset, dataloader, tokenizer, opt):
    rouge_score = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"), use_stemmer=True).to(torch.device("cuda", 0)) # select score here
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    rouge_scores_sel = []
    rouge_scores_gen = []

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
                example = dataset.get_example(idx[k])
                gold = example['target']
                score = rouge_score(ans, gold)
                rouge_scores_gen.append(score)
                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')

            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                result = {
                    'rouge1_gen': np.mean([r['rouge1_fmeasure'] for r in rouge_scores_gen]),
                    'rouge2_gen': np.mean([r['rouge2_fmeasure'] for r in rouge_scores_gen]),
                    'rougeL_gen': np.mean([r['rougeL_fmeasure'] for r in rouge_scores_gen]),
                }
                if opt.use_aux_loss:
                    result.update({
                        'rouge1_sel': np.mean([r[0] for r in rouge_scores_sel]),
                        'rouge2_sel': np.mean([r[1] for r in rouge_scores_sel]),
                        'rougeL_sel': np.mean([r[2] for r in rouge_scores_sel]),
                    })
                for k, v in result.items():
                    log += f' |\n {k} = {v:.3f}'
                logger.warning(log)
    log = f'Process rank:{opt.global_rank}, final result'
    result = {
        'rouge1_gen': np.mean([r['rouge1_fmeasure'] for r in rouge_scores_gen]),
        'rouge2_gen': np.mean([r['rouge2_fmeasure'] for r in rouge_scores_gen]),
        'rougeL_gen': np.mean([r['rougeL_fmeasure'] for r in rouge_scores_gen]),
    }
    if opt.use_aux_loss:
        result.update({
            'rouge1_sel': np.mean([r[0] for r in rouge_scores_sel]),
            'rouge2_sel': np.mean([r[1] for r in rouge_scores_sel]),
            'rougeL_sel': np.mean([r[2] for r in rouge_scores_sel]),
        })
    for k, v in result.items():
        log += f' |\n {k} = {v:.3f}'
    logger.warning(log)
    # sync across processes
    if opt.is_distributed:
        torch.distributed.barrier()
    return result


if __name__ == "__main__":
    options = Options()
    options.add_eval_options()
    opt = options.parse()
    src.dualfid.slurm.init_distributed_mode(opt)
    src.dualfid.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.dualfid.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    model_name = opt.model_type + '-' + opt.model_size
    if opt.model_type == 't5':
        model_name = "t5-" + opt.model_size
        model_class = src.dualfid.model.FiDT5
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.dualfid.data.FiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == 'dualt5':
        model_name = "t5-" + opt.model_size
        model_class = src.dualfid.model.DualFiDT5
        tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.dualfid.data.DualFiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == 'bart':
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.dualfid.model.FiDBART
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.dualfid.data.FiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)
    elif opt.model_type == 'dualbart':
        model_name = "facebook/bart-" + opt.model_size
        model_class = src.dualfid.model.DualFiDBART
        tokenizer = transformers.BartTokenizer.from_pretrained(model_name, return_dict=False)
        collator_function = src.dualfid.data.DualFiDCollator(opt.source_maxlength, tokenizer, opt.candidate_maxlength)

    else:
        raise NotImplementedError

    eval_examples = src.dualfid.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
    )
    eval_dataset = src.dualfid.data.Dataset(
        eval_examples,
        opt.n_candidates,
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=20,
        collate_fn=collator_function
    )

    # load the model from the checkpoint
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)
    opt.use_aux_loss = model.config.use_aux_loss
    opt.n_tasks = model.config.n_tasks


    logger.info("Start eval")
    evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.dualfid.util.write_output(glob_path, write_path)

