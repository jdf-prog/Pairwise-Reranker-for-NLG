import torch
import wandb
import torch.nn as nn
import numpy as np
import os
import wandb
import logging
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import EvalPrediction
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
from dualfid.curriculum import CurriculumSampler

logger = logging.getLogger(__name__)

class RerankerTrainer(Trainer):
    def evaluate(
        self,
        **kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        if self.is_world_process_zero():
            wandb.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str] = None, **kwargs):
        if self.is_world_process_zero():
            super().save_model(output_dir, **kwargs)
            model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(model.args, os.path.join(output_dir, "config.bin"))

    def _get_train_sampler(self):
        if self.args.curriculum_learning:
            sampler = CurriculumSampler(self.train_dataset, self.args.num_curriculum, self.args.curriculum_size)
            logger.info("****** Using curriculum sampler ******")
            logger.info("  Number of curriculum: {}".format(sampler.num_curriculum))
            logger.info("  Curriculum size: {}".format(sampler.curriculum_size))
            return sampler

        else:
            return super()._get_train_sampler()

class FiDTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.model.config.use_aux_loss:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                _, aux_loss = self.model.module.compute_auxiliary_loss(input["scores"])
            else:
                _, aux_loss = self.model.compute_auxiliary_loss(input["scores"])
            loss += aux_loss

        return (loss, outputs) if return_outputs else loss

# def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
#     preds, labels = eval_pred # pred_scores [batch_size, num_candidates], scores [batch_size, num_candidates, n_tasks]
#     pred_scores = preds
#     scores = labels
#     agg_scores = np.sum(scores, axis=-1) # aggregate scores

#     sort_indices = np.flip(np.argsort(agg_scores, axis=-1), axis=-1) # (batch_size, n_candidates), expected ranks
#     ranks = np.zeros_like(sort_indices)
#     ranks[np.arange(sort_indices.shape[0])[:, None], sort_indices] = np.arange(sort_indices.shape[-1])
#     pred_sort_indices = np.flip(np.argsort(pred_scores, axis=-1), axis=-1) # (batch_size, n_candidates), predicted ranks
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

def compute_metrics_for_crosscompare(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for the model.
    Args:

    """
    preds, labels = eval_pred # scores [batch_size, n_candidates, n_tasks]
    select_process, ranks_acc_dist, ranks_consistency_dist = preds
    scores = labels # [batch_size, n_candidates, n_tasks]
    # scores = scores[:, :-1] # debug
    mean_scores = np.mean(scores, axis=-1) # [batch_size, n_candidates]
    batch_size, n_candidates, n_tasks = scores.shape
    # compute accuracy
    accs = []
    for i in range(n_candidates - 1):
        cur_idx = select_process[:, 0, i]
        next_idx = select_process[:, 1, i]
        pred_better_idx = select_process[:, 2, i]
        # print(cur_idx)
        # print(next_idx)
        cur_scores = mean_scores[np.arange(batch_size), cur_idx]
        next_scores = mean_scores[np.arange(batch_size), next_idx]
        pred_better_scores = mean_scores[np.arange(batch_size), pred_better_idx]
        better_scores = np.maximum(cur_scores, next_scores)
        # print(cur_scores)
        # print(mean_scores[np.arange(batch_size), next_idx])
        # print(pred_better_scores)
        accs.append(np.mean((pred_better_scores == better_scores)))
        cur_idx = next_idx
    accs = np.array(accs)
    pred_best_idx = select_process[:, 2, -1]

    # metric_scores, denormalized these scores
    pred_best_scores = scores[np.arange(batch_size), pred_best_idx]
    oracle_best_scores = scores[np.arange(batch_size), np.argmax(mean_scores, axis=-1)]

    metrics = {
        "sel": {"acc": np.mean(accs)},
        "oracle": {},
        "top_beam": {},
    }
    for i in range(n_tasks):
        metrics["sel"]["metric_{}".format(i+1)] = np.mean(pred_best_scores[:, i])
        metrics["oracle"]["metric_{}".format(i+1)] = np.mean(oracle_best_scores[:, i])
        metrics["top_beam"]["metric_{}".format(i+1)] = np.mean(scores[:, 0, i])
    metrics['dev_score'] = metrics['sel']['metric_1']

    num_consistency = np.sum(ranks_consistency_dist[:,:,0], dtype=np.int32).item()
    num_in_consistency = np.sum(ranks_consistency_dist[:,:,1], dtype=np.int32).item()
    metrics['consistency_mean'] = np.mean(
        [1] * num_consistency + [0] * num_in_consistency
    )
    metrics['consistency_std'] = np.std(
        [1] * num_consistency + [0] * num_in_consistency
    )
    reduced_ranks_consistency_dist = np.sum(ranks_consistency_dist, axis=0) # [n_candidates]
    consistency_mean_per_rank = reduced_ranks_consistency_dist[:, 0] / np.clip(np.sum(reduced_ranks_consistency_dist, axis=-1), 1e-6, None)
    metrics.update({
        f"ranks_{i}_consistency": consistency_mean_per_rank[i] for i in range(len(consistency_mean_per_rank))
    })
    ranks_acc_dist = np.sum(ranks_acc_dist, axis=0) # [n_candidates, 2]
    ranks_acc_dist_sum = np.clip(np.sum(ranks_acc_dist, axis=-1), 1e-6, None) # in case of 0
    ranks_acc = ranks_acc_dist[:, 0] / ranks_acc_dist_sum
    metrics.update({
        f"rank_{i}_acc": ranks_acc[i] for i in range(len(ranks_acc))
    })
    return metrics


