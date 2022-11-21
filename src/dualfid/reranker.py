import argparse
from enum import unique
import sys
from regex import P
import transformers
import torch
import torch.nn as nn
import numpy as np
import copy
import random
import torch.nn.functional as F
import os
from dualfid.layers import (
    MoE,
)
from dualfid.loss import (
    infoNCE_loss,
    ListNet_loss,
    ListMLE_loss,
    p_ListMLE_loss,
    triplet_loss,
    triplet_loss_v2,
    triplet_simcls_loss,
    ApproxNDCG_loss
)
from copy import deepcopy
import transformers.trainer
from torch.nn import (
    MSELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
class SCR(nn.Module):
    """
        Roberta Sequence Classification Reranker

        Input format:
            [CLS] Source: <source> [SEP] Candidate: <candidate> [SEP]
        Output format:
            Using [CLS] token as the representation of the whole sequence.

        Support 3 objectives of reranking:
            2. multi-task classification (BCE loss)


        See SummaReranker, Refsum, BRIO for details
    """
    def __init__(self, pretrained_model, args, tokenizer=None):
        super(SCR, self).__init__()
        self.args = args
        self.n_tasks = self.args["n_tasks"]
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.drop_out = self.args.get("drop_out", 0.05)

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.hidden_size
        self.step = 0
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

        if 'moe' in self.loss_type.lower():
            self.bottom_hidden_size = self.hidden_size
            # shared bottom
            self.fc1 = nn.Linear(self.hidden_size, self.bottom_hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.bottom_hidden_size, self.hidden_size)
            # MoE
            self.moe = MoE(self.n_tasks, self.hidden_size, self.hidden_size, 2*self.n_tasks, self.hidden_size, k=self.n_tasks)
            # towers - one for each task
            self.towers = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(self.n_tasks)])
            self.sigmoid = nn.Sigmoid()
        else:
            self.regression_layer = nn.Sequential(
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 2* self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_out),
                nn.Linear(2 * self.hidden_size, self.n_tasks),
            )
            # self.regression_layer = nn.Sequential(
            #     nn.Dropout(self.drop_out),
            #     nn.Linear(self.hidden_size, self.hidden_size),
            #     nn.Tanh(),
            #     nn.Dropout(self.drop_out),
            #     nn.Linear(self.hidden_size, self.n_tasks),
            # )

    def _forawrd(self, input_ids, attention_mask):
        """
            SummareReranker
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Return:
            preds: [batch_size, n_tasks]
            aus_loss: float
        """
        _, seq_len = input_ids.shape
        # encoding source
        to_model_input_ids = input_ids.view(-1, seq_len)
        to_model_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=to_model_input_ids,
            attention_mask=to_model_attention_mask,
            output_hidden_states = True
        )
        encs = outputs["last_hidden_state"][:, 0, :] # [batch_size * n_candidates, hidden_size]
        if "moe" in self.loss_type.lower():
            # shared bottom
            encs = self.fc2(self.relu(self.fc1(encs)))
            # MoE
            moe_preds, aux_loss = self.moe(encs, train = self.training, collect_gates = not(self.training))
            # go to towers for different tasks
            pred_scores = torch.cat([
                tower(moe_pred) for moe_pred, tower in zip(moe_preds, self.towers)
            ], dim=-1)
            return pred_scores, aux_loss
        else:
            return self.regression_layer(encs), torch.tensor(0.0).to(encs.device)

    def forward(self, input_ids, attention_mask, target_ids, target_attention_mask, scores):
        """
        Args:
            input_ids: [batch_size, n_candidates, seq_len]
            attention_mask: [batch_size, n_candidates, seq_len]
            target_ids: [batch_size, seq_len]
            target_attention_mask: [batch_size, seq_len]
            scores: [batch_size, n_candidates, n_task]
        """
        self.step += 1
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(input_ids.device)
        if self.training:
            # sub sampling candidates if needed
            batch_size, n_candidates, seq_len = input_ids.shape
            selected_idx = sub_sampling(
                self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores
            )
            input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]

        # compute pred scores
        batch_size, n_candidates, seq_len = input_ids.shape
        pred_scores, aux_loss = self._forawrd(input_ids.view(-1, seq_len), attention_mask.view(-1, seq_len))
        pred_scores = pred_scores.reshape(batch_size, n_candidates, -1) # [batch_size, n_candidates, n_tasks]

        # transpose scores and labels to let the last dim be the number of candidates
        scores = scores.transpose(1, 2).reshape(-1, n_candidates)
        labels = labels.transpose(1, 2).reshape(-1, n_candidates)
        pred_scores = pred_scores.transpose(1, 2).reshape(-1, n_candidates) # [batch_size * n_tasks, n_candidates]
        # compute loss
        if self.loss_type == "MoE_BCE":
            loss = F.binary_cross_entropy_with_logits(pred_scores, labels)
        elif self.loss_type == "BCE":
            loss = F.binary_cross_entropy_with_logits(pred_scores, scores)
        elif self.loss_type == "ListMLE":
            loss = ListMLE_loss(pred_scores, scores)
        elif self.loss_type == "ListNet":
            loss = ListNet_loss(pred_scores, scores)
        elif self.loss_type == "p_ListMLE":
            loss = p_ListMLE_loss(pred_scores, scores)
        elif self.loss_type == "triplet":
            loss = triplet_loss(pred_scores, scores)
        elif self.loss_type == "triplet_v2":
            loss = triplet_loss_v2(pred_scores, scores)
        elif self.loss_type == "triplet_simcls":
            target_pred_scores = self._forawrd(target_ids, target_attention_mask)[0] # [batch_size, n_tasks]
            loss = triplet_simcls_loss(pred_scores, target_pred_scores, scores)
        elif self.loss_type == "ApproxNDCG":
            loss = ApproxNDCG_loss(pred_scores, scores)
        elif "ranknet" in self.loss_type.lower():
            dif_pred_scores = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2)
            dif_pred_scores = 1 / (1 + torch.exp(-dif_pred_scores))
            dif_scores = scores.unsqueeze(1) - scores.unsqueeze(2)
            dif_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
            dif_labels = torch.where(dif_scores == 0, torch.ones_like(dif_scores) * 0.5, dif_labels)
            loss = -(dif_labels * torch.log(dif_pred_scores) + (1 - dif_labels) * torch.log(1 - dif_pred_scores)).mean()
        else:
            raise ValueError("Unknown loss type: {}".format(self.loss_type))


        loss += aux_loss
        # return loss and logits
        pred_scores = pred_scores.reshape(batch_size, -1, n_candidates).transpose(1, 2) # [batch_size, n_candidates, n_tasks]
        pred_scores = torch.mean(pred_scores, dim=-1).detach().reshape(batch_size, n_candidates)
        pred_scores = self.sigmoid(pred_scores)
        outputs = {
            "loss": loss,
            "preds": pred_scores,
        }
        return outputs

class DualReranker(nn.Module):
    """
        Dual Encoder Reranker
        Using Roberta as backbone.

        Input format:
            source encoder: [CLS] <source>
            candidate encoder: [CLS] <candiate>
        Output formate:
            Using [CLS] embedding to do rank according

        with the similarity function as follows:
            1. dot product (DP)
            2. L2 distance (L2)
            3. negative log likelihood base on softmax (NLL)
            4. cosine similarity (Cos)

        Using Loss function
            1. InfoNCE from SimCLR (Contrastive)
            2. ListMLE (Liswise ranking)
            3. MoCo (momentum contrastive)
            4. BYOL (bootstrap your own latent)
            5. Barlow Twins

        See DPR for details
    """
    def __init__(self, pretrained_model, args, tokenizer=None):
        super(DualReranker, self).__init__()
        self.args = args
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]

        # LM
        self.source_encoder = pretrained_model
        self.candidate_encoder = deepcopy(pretrained_model)
        # self.candidate_encoder = pretrained_model
        self.hidden_size = self.source_encoder.config.hidden_size
        self.step = 0
        self.tokenizer = tokenizer

    def _forward(self,
        source_ids,
        source_attention_mask,
        target_ids,
        target_attention_mask,
        candidate_ids,
        candidate_attention_mask
    ):
        """
            Compute scores for each candidate
        Args:
            source_ids: [batch_size, source_len]
            source_attention_mask: [batch_size, source_len]
            candidate_ids: [batch_size, n_candidates, candidate_len]
            candidate_attention_mask: [batch_size, n_candidates, candidate_len]
        Returns:
            scores: [batch_size, n_candidates]
            target_scores: [batch_size]
        """

        batch_size, n_candidates, candidate_seq_len = candidate_ids.shape
        _, source_seq_len = source_ids.shape

        source_ids = source_ids.view(-1, source_seq_len)
        source_attention_mask = source_attention_mask.view(-1, source_seq_len)
        candidate_ids = candidate_ids.view(-1, candidate_seq_len)
        candidate_attention_mask = candidate_attention_mask.view(-1, candidate_seq_len)

        source_encs = self.source_encoder(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            output_hidden_states = True
        )["last_hidden_state"][:, 0, :].reshape(batch_size, 1, -1) # [batch_size, 1, hidden_size]
        candidate_encs = self.candidate_encoder(
            input_ids=candidate_ids,
            attention_mask=candidate_attention_mask,
            output_hidden_states = True
        )["last_hidden_state"][:, 0, :].reshape(batch_size, n_candidates, -1) # [batch_size, n_candidates, hidden_size]

        # compute Cosine Similarity
        source_encs = F.normalize(source_encs, dim=-1)
        candidate_encs = F.normalize(candidate_encs, dim=-1)
        sim_mat = torch.matmul(source_encs, candidate_encs.transpose(1, 2)).squeeze(1) # [batch_size, n_candidates]

        target_encs = self.candidate_encoder(
            input_ids=target_ids,
            attention_mask=target_attention_mask,
            output_hidden_states = True
        )["last_hidden_state"][:, 0, :].reshape(batch_size, 1, -1) # [batch_size, 1, hidden_size]
        target_sim_mat = torch.matmul(source_encs, target_encs.transpose(1, 2)).squeeze()
        return sim_mat, target_sim_mat

    def forward(
        self,
        source_ids,
        source_attention_mask,
        target_ids,
        target_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores):
        """
        Args:
            source_ids: [batch_size, seq_len]
            source_attention_mask: [batch_size, seq_len]
            candidate_ids: [batch_size, n_candidates, seq_len]
            candidate_attention_mask: [batch_size, n_candidates, seq_len]
            scores: [batch_size, n_candidates, n_task]
        """
        self.step += 1
        labels = torch.eq(
            torch.sum(scores, dim=-1),
            torch.max(torch.sum(scores, dim=-1), dim=1, keepdim=True)[0]
        ).float().to(source_ids.device) # [batch_size, n_candidates]
        # subsampling
        if self.training:
            batch_size, n_candidates, seq_len = candidate_ids.shape
            selected_idx = sub_sampling(self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores)
            input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]

        sim_mat, target_sim_mat = self._forward(
            source_ids, source_attention_mask,
            target_ids, target_attention_mask,
            candidate_ids, candidate_attention_mask)

        sum_scores = torch.sum(scores, dim=-1) # [batch_size, n_candidates]
        if self.loss_type == "BCE":
            loss = F.binary_cross_entropy_with_logits(sim_mat, labels)
        elif self.loss_type == "infoNCE":
            loss = infoNCE_loss(sim_mat, labels)
        elif self.loss_type == "ListMLE":
            loss = ListMLE_loss(sim_mat, sum_scores)
        elif self.loss_type == "ListNet":
            loss = ListNet_loss(sim_mat, sum_scores)
        elif self.loss_type == "p_ListMLE":
            loss = p_ListMLE_loss(sim_mat, sum_scores)
        elif self.loss_type == "triplet":
            loss = triplet_loss(sim_mat, sum_scores)
        elif self.loss_type == "triplet_v2":
            loss = triplet_loss_v2(sim_mat, sum_scores)
        elif self.loss_type == "triplet_simcls":
            loss = triplet_simcls_loss(sim_mat, target_sim_mat, sum_scores)
        else:
            raise ValueError("Loss type not supported")

        outputs = {
            "loss": loss,
            "preds": sim_mat,
        }
        return outputs

class CrossCompareReranker(nn.Module):
    """
        Cross Encoder Compare Reranker (Cross encoder version of Dual Encoder)
        Using Roberta as backbone

        Given a source text and 2 generated candidates,
        this ranker will compare the 2 candidates and give the better one by
        doing cross attention between query and 2 candidates .

        Input format:
            [CLS] source: <source> [SEP] candidate1: <candidate1> [SEP] candidate2: <candidate2> [SEP]
        Output format:
            the embeddings of the prompt 'source', 'candidate1', 'candidate2'

    """
    def __init__(self, pretrained_model, args, tokenizer):
        super(CrossCompareReranker, self).__init__()
        self.args = args
        self.n_tasks = self.args["n_tasks"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.loss_type = self.args["loss_type"]
        self.drop_out = self.args.get("drop_out", 0.05)

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.config.hidden_size
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id

        self.tokenizer = tokenizer

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, 1),
        )


        # parameters for dynamic epochs
        self.args['training_steps'] = self.args.get('training_steps', 0)
        self.args['training_data_size'] = self.args.get('training_data_size', 0)
        self.args['n_candidates'] = self.args.get('n_candidates', -1)

    def compute_loss(self, source_encs, cand1_encs, cand2_encs, left_scores, right_scores):
        device = source_encs.device
        loss = torch.tensor(0.0).to(source_encs.device)
        left_sim = F.cosine_similarity(source_encs, cand1_encs)
        right_sim = F.cosine_similarity(source_encs, cand2_encs)
        preds = left_sim - right_sim
        dif_scores = left_scores - right_scores
        if self.loss_type == "triplet":
            dif_scores_sign = torch.sign(dif_scores)
            cls_loss = torch.max(torch.zeros_like(preds), torch.abs(dif_scores) - dif_scores_sign * preds).mean()
            # add MSE loss
            mse_loss = F.mse_loss(left_sim, left_scores)
            mse_loss += F.mse_loss(right_sim, right_scores)
            cls_loss += mse_loss
        elif self.loss_type == "BCE":
            left_labels = (dif_scores > 0).float()
            right_labels = (dif_scores < 0).float()
            cls_loss = torch.tensor(0.0, device=device)
            cls_loss += F.binary_cross_entropy_with_logits(left_sim, left_labels)
            cls_loss += F.binary_cross_entropy_with_logits(right_sim, right_labels)
            cls_loss /= 2
        elif self.loss_type == "MSE":
            cls_loss = F.mse_loss(preds, dif_scores)
        elif self.loss_type == "ranknet":
            source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
            source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
            source_cand1_scores = self.head_layer(source_cand1_encs)
            source_cand2_scores = self.head_layer(source_cand2_encs)
            dif_pred_scores = source_cand1_scores - source_cand2_scores
            dif_pred_scores = 1 / (1 + torch.exp(-dif_pred_scores))
            dif_scores = left_scores - right_scores
            dif_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
            dif_labels = torch.where(dif_scores == 0, torch.zeros_like(dif_scores)*0.5, dif_labels)
            cls_loss = -(dif_labels * torch.log(dif_pred_scores) + (1 - dif_labels) * torch.log(1 - dif_pred_scores))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        loss += cls_loss
        return loss, preds

    def _forward(
        self,
        input_ids,
        attention_mask,
        scores,
        reduce="mean"
    ):
        """
            Compute scores for each candidate pairs
        Args:
            input_ids: [*, seq_len]
            attention_mask: [*, seq_len]
            scores: [*, 2], left scores, right scores
        Returns:
            pred_probs: [*]
        """
        device = input_ids.device
        original_shape = input_ids.shape[:-1]
        seq_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        scores = scores.view(-1, 2)
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        sep_token_idx = input_ids.eq(self.sep_token_id)
        assert sep_token_idx.sum(-1).eq(3).all(), sep_token_idx.sum(-1)
        sep_token_idx = sep_token_idx.nonzero()[:, 1].reshape(-1, 3)
        encs = outputs.last_hidden_state
        # get the special token <source> and <candidate>
        loss = torch.tensor(0.0, device=device)
        if reduce=="special":
            source_encs = encs[:, 1, :]
            cand1_encs = encs[torch.arange(encs.shape[0]), sep_token_idx[:, 0]+1, :]
            cand2_encs = encs[torch.arange(encs.shape[0]), sep_token_idx[:, 1]+1, :]
        elif reduce=="mean":
            source_encs = []
            cand1_encs = []
            cand2_encs = []
            for i in range(encs.shape[0]):
                source_encs.append(torch.mean(encs[i, 1:sep_token_idx[i, 0], :], dim=0))
                cand1_encs.append(torch.mean(encs[i, sep_token_idx[i, 0]+1:sep_token_idx[i, 1], :], dim=0))
                cand2_encs.append(torch.mean(encs[i, sep_token_idx[i, 1]+1:sep_token_idx[i, 2], :], dim=0))
            source_encs = torch.stack(source_encs, dim=0)
            cand1_encs = torch.stack(cand1_encs, dim=0)
            cand2_encs = torch.stack(cand2_encs, dim=0)
        else:
            raise ValueError(f"Unknown reduce type: {reduce}")


        # get the exclusive encodings of cand1 and cand2
        loss, pred_probs = self.compute_loss(source_encs, cand1_encs, cand2_encs, scores[:, 0], scores[:, 1])

        pred_probs = pred_probs.view(*original_shape)
        outputs = {
            "loss": loss,
            "preds": pred_probs,
            "source_encs": source_encs,
            "cand1_encs": cand1_encs,
            "cand2_encs": cand2_encs,
        }
        return outputs

    def sampling(self, scores):
        """
        Args:
            scores: [batch_size, n_candidates]
            n_pair: int
            device: torch.device
        """
        batch_size, n_candidates = scores.shape
        n_pair = min(self.num_pos, self.num_neg)
        device = scores.device

        sorted_idx = torch.argsort(scores, dim=1, descending=True) # [batch_size, n_candidates]
        # NOTE: different sampling strategy
        if self.sub_sampling_mode == "top_bottom":
            # 1. top bottom sampling
            # self.args['training_steps'] += batch_size
            # cur_stage = self.args['training_steps'] // (self.args['training_data_size'] // self.args['n_candidates'] * 2)
            # dif_rank = n_candidates - cur_stage
            # rand_int = random.randint(0, cur_stage)
            # sub_sorted_idx = sorted_idx[:, rand_int:rand_int+dif_rank]
            # pos_idx = sub_sorted_idx[:, :n_pair]
            # neg_idx = sub_sorted_idx[:, -n_pair:]
            pos_idx = sorted_idx[:, :n_pair]
            neg_idx = sorted_idx[:, -n_pair:]
        elif self.sub_sampling_mode == "random":
            # 2. random sampling
            pos_idx = torch.randint(0, n_candidates, (batch_size, n_pair), device=device)
            neg_idx = torch.randint(0, n_candidates, (batch_size, n_pair), device=device)
        elif self.sub_sampling_mode == "uniform":
            # 3. uniform sampling
            step = torch.tensor(n_candidates / (n_candidates * self.sub_sampling_ratio), dtype=torch.long)
            pos_idx = sorted_idx[:, 0:-step:step]
            neg_idx = sorted_idx[:, step::step]
            n_pair = pos_idx.shape[1]
        elif self.sub_sampling_mode == "poisson_dynamic":
            # 2. using rank dif as difficulty measure function
            self.args['training_steps'] += batch_size
            if self.args['training_steps'] > self.args['training_data_size']:
                self.args['training_steps'] = 0
            poisson_rate_update_steps = self.args['training_data_size'] // self.args['n_candidates']
            poisson_rate = self.args['training_steps'] // poisson_rate_update_steps + 1
            poisson_rate = max(poisson_rate, 1)
            # print(poisson_rate_update_steps) # debug
            # print(f"poisson_rate: {poisson_rate}") # debug
            poisson_sampling_dif_ranks = torch.clip(
                torch.poisson(torch.ones(batch_size, n_pair) * poisson_rate), 1, self.args['n_candidates'] - 1
            ).type(torch.int64).to(device)
            poisson_sampling_dif_ranks = self.args['n_candidates'] - 1 - poisson_sampling_dif_ranks
            # print(f"poisson_sampling_dif_ranks: {poisson_sampling_dif_ranks}") # debug
            pos_idx = torch.tensor([
                [random.randint(0, self.args['n_candidates'] - 1 - poisson_sampling_dif_ranks[i,j].item())
                    for j in range(n_pair)] for i in range(batch_size)
            ]).type(torch.int64).to(device)
            neg_idx = pos_idx + poisson_sampling_dif_ranks
            # print(f"pos_idx: {pos_idx}") # debug
            # print(f"neg_idx: {neg_idx}") # debug
            # print(f"sorted_idx: {sorted_idx}") # debug
            pos_idx = sorted_idx.gather(1, pos_idx)
            neg_idx = sorted_idx.gather(1, neg_idx)
            # print(f"pos_idx: {pos_idx}") # debug
            # print(f"neg_idx: {neg_idx}") # debug
            # TODO: select unique pairs
        else:
            raise ValueError(f"Unknown sampling mode: {self.sub_sampling_mode}")

        shuffle_flag = torch.rand(batch_size, n_pair, device=device) < 0.5
        left_idx = torch.where(shuffle_flag, neg_idx, pos_idx)
        right_idx = torch.where(shuffle_flag, pos_idx, neg_idx)
        return left_idx, right_idx

    def forward(
        self,
        candidate_pair_ids,
        candidate_pair_attention_mask,
        scores,
    ):
        """
            Compute scores for each candidate
        Args:
            passing_in as group:
                candidate_pair_ids: [batch_size, n_candidates, n_candidates, candidate_len]
                candidate_pair_attention_mask: [batch_size, n_candidates, n_candidates, candidate_len]
                scores: [batch_size, n_candidates, n_tasks]
            passing in as individual:
                candidate_pair_ids: [batch_size, candidate_len]
                candidate_pair_attention_mask: [batch_size, candidate_len]
                scores: [batch_size, 2, n_tasks] (left, right)
        """
        device = candidate_pair_ids.device
        outputs = {}
        if candidate_pair_ids.dim() == 2:
            # passing in as individual
            outputs = self._forward(
                candidate_pair_ids,
                candidate_pair_attention_mask,
                scores.mean(-1)
            )

        elif candidate_pair_ids.dim() == 4:

            # passing in as group
            if self.training:
                # subsampling
                batch_size, n_candidates, n_tasks = scores.shape
                if self.args['n_candidates'] == -1:
                    self.args['n_candidates'] = n_candidates
                scores = scores.mean(dim=-1)


                left_idx, right_idx = self.sampling(scores)
                batch_idx = torch.arange(batch_size).unsqueeze(1)

                candidate_pair_ids = candidate_pair_ids[batch_idx, left_idx, right_idx] # [batch_size, n_pair, candidate_len]
                candidate_pair_attention_mask = candidate_pair_attention_mask[batch_idx, left_idx, right_idx] # [batch_size, n_pair, candidate_len]
                left_scores = scores[batch_idx, left_idx]
                right_scores = scores[batch_idx, right_idx]
                outputs = self._forward(
                    candidate_pair_ids,
                    candidate_pair_attention_mask,
                    torch.stack([left_scores, right_scores], dim=-1)
                )

            else:
                batch_size, n_candidates, _, candidate_len = candidate_pair_ids.shape
                scores = torch.mean(scores, dim=-1) # [batch_size, n_candidates]
                sorted_idx = torch.argsort(scores, dim=1, descending=True) # [batch_size, n_candidates]
                ranks = torch.zeros_like(sorted_idx)
                ranks[torch.arange(batch_size).unsqueeze(1), sorted_idx] = torch.arange(n_candidates, device=device)

                permu = torch.randperm(n_candidates).repeat(batch_size, 1).to(device) # [batch_size, n_candidates] random

                cur_idx = permu[:, 0]
                loss = torch.tensor(0.0).to(device)
                initial_idx = cur_idx
                next_idxs = []
                better_idxs = []
                ranks_acc_dist = torch.zeros(batch_size, n_candidates, 2, device=device)
                ranks_consistency_dist = torch.zeros(batch_size, n_candidates, 2, device=device)
                for i in range(1, n_candidates):
                    next_idx = permu[:, i] # [batch_size]

                    to_model_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                    to_model_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                    to_model_ids = to_model_ids.view(batch_size, candidate_len)
                    to_model_attention_mask = to_model_attention_mask.view(batch_size, candidate_len)
                    to_model_scores = torch.stack(
                        [scores[torch.arange(batch_size), cur_idx], scores[torch.arange(batch_size), next_idx]],
                    dim=-1) # [batch_size, 2]
                    _outputs = self._forward(
                        to_model_ids, to_model_attention_mask, to_model_scores
                    ) # [batch_size]
                    loss += _outputs['loss']
                    preds = _outputs['preds']

                    to_model_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), next_idx.unsqueeze(1), cur_idx.unsqueeze(1), :]
                    to_model_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), next_idx.unsqueeze(1), cur_idx.unsqueeze(1), :]
                    to_model_ids = to_model_ids.view(batch_size, candidate_len)
                    to_model_attention_mask = to_model_attention_mask.view(batch_size, candidate_len)
                    to_model_scores = torch.stack(
                        [scores[torch.arange(batch_size), next_idx], scores[torch.arange(batch_size), cur_idx]],
                    dim=-1) # [batch_size, 2]
                    _outputs = self._forward(
                        to_model_ids, to_model_attention_mask, to_model_scores
                    ) # [batch_size]
                    loss += _outputs['loss']
                    preds_inv = -_outputs['preds']

                    consistency = ((preds * preds_inv) > 0).float()

                    # NOTE: compute distribution of the accuracy with different ranks
                    oracle_compare_result = scores[torch.arange(batch_size), cur_idx] - scores[torch.arange(batch_size), next_idx]
                    pair_dif_ranks = ranks[torch.arange(batch_size), cur_idx] - ranks[torch.arange(batch_size), next_idx]
                    pair_dif_ranks = torch.abs(pair_dif_ranks)
                    for bz in range(batch_size):
                        if (preds[bz]) * oracle_compare_result[bz] > 0:
                            ranks_acc_dist[bz, pair_dif_ranks[bz], 0] += 1
                        else:
                            ranks_acc_dist[bz, pair_dif_ranks[bz], 1] += 1
                        if (preds_inv[bz]) * oracle_compare_result[bz] > 0:
                            ranks_acc_dist[bz, pair_dif_ranks[bz], 0] += 1
                        else:
                            ranks_acc_dist[bz, pair_dif_ranks[bz], 1] += 1
                        if consistency[bz] > 0:
                            ranks_consistency_dist[bz, pair_dif_ranks[bz], 0] += 1
                        else:
                            ranks_consistency_dist[bz, pair_dif_ranks[bz], 1] += 1

                    better_idx = torch.where(preds + preds_inv > 0, cur_idx, next_idx)
                    # better_idx = torch.where(cand1_sim >= cand2_sim, cur_idx, next_idx)
                    better_idxs.append(better_idx)
                    next_idxs.append(next_idx)
                    cur_idx = better_idx
                outputs = {}
                outputs['loss'] = loss
                outputs["select_process"] = []
                outputs["select_process"].append(torch.stack([initial_idx] + better_idxs[:-1], dim=1))
                outputs["select_process"].append(torch.stack(next_idxs, dim=1))
                outputs["select_process"].append(torch.stack(better_idxs, dim=1))
                outputs["select_process"] = torch.stack(outputs["select_process"], dim=1) # [batch_size, 3, n_candidates]
                outputs["ranks_acc_dist"] = ranks_acc_dist # [batch_size, n_candidates, 2]
                outputs["ranks_consistency_dist"] = ranks_consistency_dist # [batch_size, n_candidates, 2]
                assert outputs["select_process"].shape == (batch_size, 3, n_candidates-1), outputs["select_process"].shape

        return outputs

class DualCompareReranker(nn.Module):
    """
        Cross Encoder Compare Reranker (Cross encoder version of Dual Encoder)
        Using Roberta as backbone

        Given a source text and 2 generated candidates,
        this ranker will compare the 2 candidates and give the better one by
        doing cross attention between query and 2 candidates .

        Input format:
            [CLS] source: <source> [SEP] candidate1: <candidate1> [SEP] candidate2: <candidate2> [SEP]
        Output format:
            the embeddings of the prompt 'source', 'candidate1', 'candidate2'

    """
    def __init__(self, pretrained_model, args, tokenizer=None):
        super(DualCompareReranker, self).__init__()
        self.args = args
        self.n_tasks = self.args["n_tasks"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.loss_type = self.args["loss_type"]
        self.drop_out = self.args.get("drop_out", 0.05)


        # LM
        self.source_encoder = pretrained_model
        self.candidate_encoder = deepcopy(pretrained_model)
        self.sep_token = pretrained_model.config.eos_token_id

        self.tokenizer = tokenizer

    def _encode_source(self, source_ids, source_attention_mask):
        """
            Encode source text
        Args:
            source_ids: [batch_size, source_len]
            source_attention_mask: [batch_size, source_len]
        Returns:

        """
        batch_size, source_len = source_ids.shape
        last_hidden_states = self.source_encoder(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        ).last_hidden_state # [batch_size, source_len, hidden_size]
        source_sep_idx = (source_ids == self.tokenizer.sep_token_id)
        assert all(source_sep_idx.sum(1) == 1), source_sep_idx.sum(1)
        source_sep_idx = source_sep_idx.nonzero()[:, 1] # [batch_size]

        # mean pooling
        source_encs = []
        for i in range(batch_size):
            source_token_encs = last_hidden_states[i, 1:source_sep_idx[i], :] # [source_len, hidden_size]
            source_enc = torch.mean(source_token_encs, dim=0) # [hidden_size]
            source_encs.append(source_enc)
        source_encs = torch.stack(source_encs, dim=0) # [batch_size, hidden_size]
        return source_encs

    def _encode_candidates(
        self,
        candidate_pair_ids,
        candidate_pair_attention_mask,
    ):
        """
            Compute scores for each candidate pairs
        Args:
            source_ids: [batch_size, source_len]
            source_attention_mask: [batch_size, source_len]
            candidate_pair_ids: [batch_size, n_pair, candidate_len]
            candidate_pair_attention_mask: [batch_size, n_pair, candidate_len]
        Returns:
            pred_probs: [*]
        """
        candidate_len = candidate_pair_ids.shape[-1]
        original_shape = candidate_pair_ids.shape[:-1]
        candidate_pair_ids = candidate_pair_ids.view(-1, candidate_len)
        candidate_pair_attention_mask = candidate_pair_attention_mask.view(-1, candidate_len)

        outputs = self.candidate_encoder(
            input_ids=candidate_pair_ids,
            attention_mask=candidate_pair_attention_mask,
            output_hidden_states=True,
        )
        sep_token_idx = candidate_pair_ids.eq(self.sep_token)
        assert sep_token_idx.sum(-1).eq(2).all(), sep_token_idx.sum(-1)
        sep_token_idx = sep_token_idx.nonzero()[:, 1].reshape(-1, 2)

        # compute candidate encs
        encs = outputs.last_hidden_state
        cand1_encs = []
        cand2_encs = []
        random = torch.randint(0, 100, (1,)).item()
        for i in range(encs.shape[0]):
            # compute per token weight
            cand1_enc = encs[i, 1:sep_token_idx[i, 0]] # [cand1_len, hidden_size]
            cand2_enc = encs[i, sep_token_idx[i, 0]+1:sep_token_idx[i, 1]] # [cand2_len, hidden_size]
            sim_mat = torch.einsum("ij,kj->ik", F.normalize(cand1_enc, dim=-1), F.normalize(cand2_enc, dim=-1)) # [cand1_len, cand2_len]
            cand1_weight = (1-(sim_mat).max(1)[0])
            cand2_weight = (1-(sim_mat).max(0)[0])
            cand1_weight = cand1_weight / cand1_weight.sum()
            cand2_weight = cand2_weight / cand2_weight.sum()
            output_heatmap = False # debug
            if output_heatmap:
                cand1_ids = candidate_pair_ids[i, 1:sep_token_idx[i, 0]] # [cand1_len]
                cand2_ids = candidate_pair_ids[i, sep_token_idx[i, 0]+1:sep_token_idx[i, 1]] # [cand2_len]
                cand1_tokens = self.tokenizer.convert_ids_to_tokens(cand1_ids.tolist())
                cand2_tokens = self.tokenizer.convert_ids_to_tokens(cand2_ids.tolist())
                cand1_text = self.tokenizer.decode(cand1_ids.tolist())
                cand2_text = self.tokenizer.decode(cand2_ids.tolist())
                sim_mat_numpy = sim_mat.detach().cpu().numpy().round(3)
                df = pd.DataFrame(sim_mat_numpy, index=cand1_tokens, columns=cand2_tokens)
                plt.figure(figsize=(len(cand2_tokens)*0.75, len(cand1_tokens)*0.75), dpi=100)
                plt.title(f"{cand1_text}\n{cand2_text}")
                sns.heatmap(df, annot=True, fmt=".3f", cmap="Blues").get_figure().savefig(f"./pics/sim_mat_{i}_{random}.png")
            # cand1_encs.append((cand1_enc * cand1_weight.unsqueeze(1)).sum(0))
            # cand2_encs.append((cand2_enc * cand2_weight.unsqueeze(1)).sum(0))
            cand1_encs.append(cand1_enc.mean(0)) # debug
            cand2_encs.append(cand2_enc.mean(0)) # debug

        cand1_encs = torch.stack(cand1_encs, dim=0)
        cand2_encs = torch.stack(cand2_encs, dim=0)
        cand1_encs = cand1_encs.view(*original_shape, -1) # [batch_size, n_pair, hidden_size]
        cand2_encs = cand2_encs.view(*original_shape, -1) # [batch_size, n_pair, hidden_size]
        return cand1_encs, cand2_encs


    def forward(
        self,
        source_ids,
        source_attention_mask,
        candidate_pair_ids,
        candidate_pair_attention_mask,
        candidate_target_ids,
        candidate_target_attention_mask,
        scores,
        cand_target_dif_scores
    ):
        """
            Compute scores for each candidate
        Args:
            source_ids: [batch_size, source_len]
            source_attention_mask: [batch_size, source_len]
            candidate_pair_ids: [batch_size, n_candidates, n_candidates, candidate_len]
            candidate_pair_attention_mask: [batch_size, n_candidates, n_candidates, candidate_len]
            candidate_target_ids: [batch_size, n_candidates, candidate_len]
            candidate_target_attention_mask: [batch_size, n_candidates, candidate_len]
            scores: [batch_size, n_candidates, n_tasks]
            cand_target_dif_scores: [batch_size, n_candidates, n_tasks]
        """
        device = source_ids.device

        if self.training:
            # subsampling
            batch_size, n_candidates, n_tasks = scores.shape
            scores = scores.sum(dim=-1)

            sorted_idx = torch.argsort(scores, dim=1, descending=True) # [batch_size, n_candidates]
            n_pair = min(self.num_pos, self.num_neg)
            pos_idx = sorted_idx[:, :n_pair]
            neg_idx = sorted_idx[:, -n_pair:]
            shuffle_flag = torch.rand(batch_size, n_pair, device=device) < 0.5
            left_idx = torch.where(shuffle_flag, neg_idx, pos_idx)
            right_idx = torch.where(shuffle_flag, pos_idx, neg_idx)
            candidate_pair_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), left_idx, right_idx] # [batch_size, n_pair, candidate_len]
            candidate_pair_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), left_idx, right_idx] # [batch_size, n_pair, candidate_len]

            dif_scores = scores[torch.arange(batch_size).unsqueeze(1), left_idx] - scores[torch.arange(batch_size).unsqueeze(1), right_idx]
            left_labels = (dif_scores > 0).float()
            right_labels = (dif_scores < 0).float()

            source_encs = self._encode_source(source_ids, source_attention_mask) # [batch_size, hidden_size]
            cand1_encs, cand2_encs = self._encode_candidates(candidate_pair_ids, candidate_pair_attention_mask) # [batch_size, n_pair, hidden_size]

            # commpute similarity
            expanded_source_encs = source_encs.unsqueeze(1).expand(-1, n_pair, -1).reshape(batch_size * n_pair, -1) # [batch_size * n_pair, hidden_size]
            left_sim = F.cosine_similarity(expanded_source_encs, cand1_encs, dim=-1).view(batch_size, n_pair) # [batch_size, n_pair]
            right_sim = F.cosine_similarity(expanded_source_encs, cand2_encs, dim=-1).view(batch_size, n_pair) # [batch_size, n_pair]
            # # compute BCE loss
            loss = torch.tensor(0.0, device=device)
            loss += F.binary_cross_entropy_with_logits(left_sim, left_labels)
            loss += F.binary_cross_entropy_with_logits(right_sim, right_labels)
            loss /= 2

            outputs = {
                "loss": loss,
            }
        else:
            batch_size, n_candidates, _, candidate_len = candidate_pair_ids.shape
            scores = torch.mean(scores, dim=-1) # [batch_size, n_candidates]
            permu = torch.randperm(n_candidates).repeat(batch_size, 1).to(device) # [batch_size, n_candidates]
            cur_idx = permu[:, 0]
            initial_idx = cur_idx
            next_idxs = []
            better_idxs = []

            source_encs = self._encode_source(source_ids, source_attention_mask) # [batch_size, hidden_size]

            for i in range(1, n_candidates):
                next_idx = permu[:, i]
                to_model_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                to_model_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                to_model_ids = to_model_ids.view(batch_size, candidate_len)
                to_model_attention_mask = to_model_attention_mask.view(batch_size, candidate_len)
                cand1_encs, cand2_encs = self._encode_candidates(
                    to_model_ids, to_model_attention_mask,
                )
                # commpute similarity
                left_sim = F.cosine_similarity(source_encs, cand1_encs, dim=-1) # [batch_size]
                right_sim = F.cosine_similarity(source_encs, cand2_encs, dim=-1) # [batch_size]

                # compute accuracy
                better_idx = torch.where(left_sim >= right_sim, cur_idx, next_idx)
                better_idxs.append(better_idx)
                next_idxs.append(next_idx)
                cur_idx = better_idx
            outputs = {
                "loss": torch.tensor(0.0, device=device),
            }
            outputs["select_process"] = []
            outputs["select_process"].append(torch.stack([initial_idx] + better_idxs[:-1], dim=1))
            outputs["select_process"].append(torch.stack(next_idxs, dim=1))
            outputs["select_process"].append(torch.stack(better_idxs, dim=1))
            outputs["select_process"] = torch.stack(outputs["select_process"], dim=1) # [batch_size, 3, n_candidates]
            assert outputs["select_process"].shape == (batch_size, 3, n_candidates-1), outputs["select_process"].shape
        return outputs

def sub_sampling(mode, num_pos, num_neg, ratio, scores):
    """
    Args:
        mode: sub sampling mode
        num_pos: number of positive samples
        num_neg: number of negative samples
        ratio: ratio of positive samples
        scores: [batch_size, candidate, n_task]

    Returns:
        selected_idx: [batch_size, n_pos+n_neg] or [batch_size, n_candidates * ratio]

    """
    batch_size, n_candidates, n_task = scores.shape

    if mode == "uniform":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=True)
        step = torch.tensor(n_candidates / (n_candidates * ratio), dtype=torch.long)
        selected_idx = sorted_idx[:, ::step]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "top":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=True)
        selected_idx = sorted_idx[:, :int(n_candidates * ratio)]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "bottom":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=False)
        selected_idx = sorted_idx[:, :int(n_candidates * ratio)]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "random":
        selected_idx = torch.stack([
            torch.randperm(n_candidates)[:int(n_candidates * ratio)] for _ in range(batch_size)
        ], dim=0) # [batch_size, n_candidates * ratio]
    elif mode in ["top_bottom", "top_random", "random_bottom"]:
        selected_idx = []
        for i in range(batch_size):
            idx = np.arange(n_candidates)
            # remove duplicate candidates, cpu
            unique_idx = []
            unique_scores = []
            for j, score in enumerate(torch.sum(scores[i], dim=-1)):
                if score not in unique_scores:
                    unique_idx.append(idx[j])
                    unique_scores.append(score.item())
            unique_idx = np.array(unique_idx)
            unique_scores = np.array(unique_scores)
            # only select a few pos and neg candidates
            sorted_idx = np.argsort(unique_scores)[::-1]

            if mode == "top_bottom":
                pos_idx = sorted_idx[:num_pos] # top
                neg_idx = sorted_idx[-num_neg:] # bottom
            elif mode == "top_random":
                pos_idx = sorted_idx[:num_pos] # top
                neg_idx = np.random.choice(sorted_idx[num_pos:], num_neg, replace=False) # random
            elif mode == "random_bottom":
                pos_idx = np.random.choice(sorted_idx[:-num_neg], num_pos, replace=False) # random
                neg_idx = sorted_idx[-num_neg:] # bottom
            else:
                raise NotImplementedError
            idx = np.concatenate([pos_idx, neg_idx])
            np.random.shuffle(idx)
            idx = unique_idx[idx]
            selected_idx.append(idx)
        selected_idx = torch.tensor(selected_idx)
    else:
        raise NotImplementedError

    return selected_idx
