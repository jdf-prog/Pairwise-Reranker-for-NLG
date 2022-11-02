import torch
import wandb
import torch.nn as nn
import numpy as np
import os
from transformers.trainer import Trainer
from transformers import EvalPrediction
from typing import Dict, List, Optional, Tuple, Union, Any
from common.evaluation import (
    eval_rouge,
    eval_bleu,
)
import wandb

class RerankerTrainer(Trainer):
    def evaluate(
        self,
        **kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        if self.is_world_process_zero():
            wandb.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        if self.is_world_process_zero():
            super().save_model(output_dir)
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(model.args, os.path.join(output_dir, "config.bin"))


class FiDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs['labels'],
        )
        loss = outputs[0]
        config = model.module.config if hasattr(model, "module") else model.config
        if config.use_aux_loss:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                _, aux_loss = model.module.compute_auxiliary_loss(inputs['scores'])
            else:
                _, aux_loss = model.compute_auxiliary_loss(inputs['scores'])
        else:
            aux_loss = 0
        loss = loss + aux_loss
        return (loss, outputs) if return_outputs else loss

# def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
#     preds, labels = eval_pred # pred_scores [batch_size, num_candidates], scores [batch_size, num_candidates, n_tasks]
#     pred_scores = preds
#     scores = labels
#     agg_scores = np.sum(scores, axis=-1) # aggregate scores

#     sort_indices = np.flip(np.argsort(agg_scores, axis=-1), axis=-1) # (batch_size, n_candidate), expected ranks
#     ranks = np.zeros_like(sort_indices)
#     ranks[np.arange(sort_indices.shape[0])[:, None], sort_indices] = np.arange(sort_indices.shape[-1])
#     pred_sort_indices = np.flip(np.argsort(pred_scores, axis=-1), axis=-1) # (batch_size, n_candidate), predicted ranks
#     pred_ranks = np.zeros_like(pred_sort_indices)
#     pred_ranks[np.arange(pred_sort_indices.shape[0])[:, None], pred_sort_indices] = np.arange(pred_sort_indices.shape[-1])

#     # compute selection scores
#     sel_idx = np.argmax(pred_scores, axis=1) # [batch_size]
#     sel_scores = scores[np.arange(scores.shape[0]), sel_idx] # [batch_size, n_task]
#     sel_ranks = ranks[np.arange(ranks.shape[0]), sel_idx] # [batch_size]
#     sel_acc = np.mean((sel_ranks == 0)) # scalar

#     # compute oracle scores for reference
#     oracle_sel_idx = np.argmax(agg_scores, axis=1) # [batch_size]
#     oracle_sel_scores = scores[np.arange(scores.shape[0]), oracle_sel_idx] # [batch_size, n_task]
#     oracle_sel_ranks = ranks[np.arange(ranks.shape[0]), oracle_sel_idx] # [batch_size]
#     oracle_sel_acc = np.mean((oracle_sel_ranks == 0)) # scalar

#     metrics = {
#         "sel": {
#             "acc": sel_acc,
#             "rank": np.mean(sel_ranks),
#             "rouge1": np.mean(sel_scores[:, 0]),
#             "rouge2": np.mean(sel_scores[:, 1]),
#             "rougeL": np.mean(sel_scores[:, 2]),
#         },
#         "oracle": {
#             "acc": oracle_sel_acc,
#             "rank": np.mean(oracle_sel_ranks),
#             "rouge1": np.mean(oracle_sel_scores[:, 0]),
#             "rouge2": np.mean(oracle_sel_scores[:, 1]),
#             "rougeL": np.mean(oracle_sel_scores[:, 2]),
#         },
#         "dev_score": np.mean(sel_scores[:, 1]), # dev score used for save checkpoint
#     }
#     return metrics

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds, labels = eval_pred # pred_scores [batch_size, n_candidates], scores [batch_size, n_candidates, n_tasks]
    select_process = preds # [batch_size, n_candidates]
    scores = labels # [batch_size, n_candidates, n_tasks]
    sum_scores = np.sum(scores, axis=-1) # [batch_size, n_candidates]
    batch_size, n_candidates, n_tasks = scores.shape
    # compute accuracy
    accs = []
    for i in range(n_candidates - 1):
        cur_idx = select_process[:, 0, i]
        next_idx = select_process[:, 1, i]
        pred_better_idx = select_process[:, 2, i]
        # print(cur_idx)
        # print(next_idx)
        cur_scores = sum_scores[np.arange(batch_size), cur_idx]
        next_scores = sum_scores[np.arange(batch_size), next_idx]
        pred_better_scores = sum_scores[np.arange(batch_size), pred_better_idx]
        better_scores = np.maximum(cur_scores, next_scores)
        # print(cur_scores)
        # print(sum_scores[np.arange(batch_size), next_idx])
        # print(pred_better_scores)
        accs.append(np.mean((pred_better_scores == better_scores)))
        cur_idx = next_idx
    accs = np.array(accs)
    pred_best_idx = select_process[:, 2, -1]
    pred_best_scores = scores[np.arange(batch_size), pred_best_idx]
    oracle_best_scores = scores[np.arange(batch_size), np.argmax(sum_scores, axis=-1)]


    metrics = {
        "sel": {
            "acc": np.mean(accs),
            "rouge1": np.mean(pred_best_scores[:, 0]),
            "rouge2": np.mean(pred_best_scores[:, 1]),
            "rougeL": np.mean(pred_best_scores[:, 2]),
        },
        "oracle": {
            "rouge1": np.mean(oracle_best_scores[:, 0]),
            "rouge2": np.mean(oracle_best_scores[:, 1]),
            "rougeL": np.mean(oracle_best_scores[:, 2]),
        },
        "dev_score": np.mean(pred_best_scores[:, 1]), # dev score used for save checkpoint
    }
    return metrics
