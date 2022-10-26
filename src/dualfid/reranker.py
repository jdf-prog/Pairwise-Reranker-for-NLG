import argparse
from enum import unique
import sys
import transformers
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
import os
from dualfid.layers import (
    MoE,
    infoNCE_loss,
    ListNet_loss,
    ListMLE_loss,
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
    def __init__(self, pretrained_model, args):
        super(SCR, self).__init__()
        self.args = args
        self.n_tasks = self.args["n_tasks"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.top_k_permutation = self.args.get("top_k_permutation", 1)
        self.drop_out = self.args.get("drop_out", 0.05)

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.hidden_size

        if self.loss_type == "BCE":
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

            self.loss = nn.BCEWithLogitsLoss()
        elif self.loss_type == "ListMLE":
            self.regression_layer = nn.Sequential(
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_out),
                nn.Linear(2 * self.hidden_size, 1),
            )
        elif self.loss_type == "ListNet":
            self.regression_layer = nn.Sequential(
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_out),
                nn.Linear(2 * self.hidden_size, 1),
            )
            self.loss = nn.CrossEntropyLoss()

    def BCE_loss(self, encs, labels):
        """
            SummareReranker
        Args:
            encs: [batch_size, n_candidate, hidden_size]
            labels: [batch_size, n_candidate, n_task]
        Return:
            loss: [1]
            preds: [batch_size, n_candidate]
        """
        batch_size, n_candidate, n_task = labels.shape
        # shared bottom
        encs = self.fc2(self.relu(self.fc1(encs))).reshape(batch_size * n_candidate, -1)
        # MoE
        moe_preds, aux_loss = self.moe(encs, train = self.training, collect_gates = not(self.training))
        # go to towers for different tasks
        pred_scores = torch.cat([
            tower(moe_pred) for moe_pred, tower in zip(moe_preds, self.towers)
        ], dim=-1).reshape(batch_size, n_candidate, n_task)
        # compute loss
        loss = self.loss(pred_scores, labels)
        loss += aux_loss
        # return loss and logits
        pred_scores = torch.mean(pred_scores, dim=-1).detach().reshape(batch_size, n_candidate)
        pred_scores = self.sigmoid(pred_scores)
        return loss, pred_scores

    def forward(self, input_ids, attention_mask, scores):
        """
        Args:
            input_ids: [batch_size, n_candidate, seq_len]
            attention_mask: [batch_size, n_candidate, seq_len]
            scores: [batch_size, n_candidate, n_task]
        """

        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(input_ids.device)
        if self.training:
            # sub sampling candidates if needed
            if self.num_pos > 0 and self.num_neg > 0:
                sub_idx, input_ids, attention_mask, scores, labels = \
                    sub_sampling(input_ids, attention_mask, scores, labels, self.num_pos, self.num_neg)

        batch_size, n_candidate, seq_len = input_ids.shape

        to_model_input_ids = input_ids.view(-1, seq_len)
        to_model_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=to_model_input_ids,
            attention_mask=to_model_attention_mask,
            output_hidden_states = True
        )
        encs = outputs["last_hidden_state"][:, 0, :] # [batch_size * n_candidate, hidden_size]

        if self.loss_type == "BCE":
            loss, pred_scores = self.BCE_loss(encs.reshape(batch_size, n_candidate, -1), labels)
        elif self.loss_type == "ListMLE":
            sum_scores = torch.sum(scores, dim=-1) # [batch_size, n_candidate]
            pred_scores = self.regression_layer(encs).reshape(batch_size, n_candidate) # [batch_size, n_candidate]
            loss = ListMLE_loss(pred_scores, sum_scores)
        elif self.loss_type == "ListNet":
            sum_scores = torch.sum(scores, dim=-1) # [batch_size, n_candidate]
            pred_scores = self.regression_layer(encs).reshape(batch_size, n_candidate) # [batch_size, n_candidate]
            loss = ListNet_loss(pred_scores, sum_scores, self.top_k_permutation)

        outputs = {
            "loss": loss,
            "preds": pred_scores,
        }
        return outputs


class T5SCR(nn.Module):
    """
        T5 Sequence Classification Reranker

        Input format:
            [CLS] Source: <source> [SEP] Candidate: <candidate> [SEP] Is this a good summary? [SEP]
        Output format:
            logits of 'true' or 'false'

        Support 3 objectives of reranking:
            1. multi-task regression (MSE loss)
            2. multi-task classification (BCE loss)
            3. learning to rank (NDCG loss)

        See paper:http://arxiv.org/abs/2003.06713 for details
    """

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
    def __init__(self, pretrained_model, args):
        super(DualReranker, self).__init__()
        self.args = args
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.top_k_permutation = self.args["top_k_permutation"]

        # LM
        self.source_encoder = pretrained_model
        self.candidate_encoder = deepcopy(pretrained_model)
        self.hidden_size = self.source_encoder.config.hidden_size


    def forward(self, source_ids, source_attention_mask, candidate_ids, candidate_attention_mask, scores):
        """
        Args:
            source_ids: [batch_size, seq_len]
            source_attention_mask: [batch_size, seq_len]
            candidate_ids: [batch_size, n_candidate, seq_len]
            candidate_attention_mask: [batch_size, n_candidate, seq_len]
            scores: [batch_size, n_candidate, n_task]
        """
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(source_ids.device)
        if self.training:
            if self.num_pos > 0 and self.num_neg > 0:
                sub_idx, candidate_ids, candidate_attention_mask, scores, labels = \
                    sub_sampling(candidate_ids, candidate_attention_mask, scores, labels, self.num_pos, self.num_neg)

        batch_size, n_candidate, candidate_seq_len = candidate_ids.shape
        _, source_seq_len = source_ids.shape
        n_task = scores.shape[-1]

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
        )["last_hidden_state"][:, 0, :].reshape(batch_size, n_candidate, -1) # [batch_size, n_candidate, hidden_size]

        # compute similarity matrix
        source_encs = F.normalize(source_encs, dim=-1)
        candidate_encs = F.normalize(candidate_encs, dim=-1)
        sim_mat = torch.matmul(source_encs, candidate_encs.transpose(1, 2)).squeeze(1) # [batch_size, n_candidate]
        sum_scores = torch.sum(scores, dim=-1).to(source_ids.device) # [batch_size, n_candidate]

        if self.loss_type == "BCE":
            loss = F.binary_cross_entropy_with_logits(sim_mat, labels)
        elif self.loss_type == "infoNCE":
            loss = infoNCE_loss(sim_mat, labels)
        elif self.loss_type == "ListMLE":
            loss = ListMLE_loss(sim_mat, sum_scores)
        elif self.loss_type == "ListNet":
            loss = ListNet_loss(sim_mat, sum_scores, self.top_k_permutation)
        else:
            raise ValueError("Loss type not supported")

        outputs = {
            "loss": loss,
            "preds": sim_mat,
        }
        return outputs

class ListWiseReranker(nn.Module):
    """
        Listwise Reranker
        Using Roberta as backbone.

        Input format:
            [CLS] Source: <source> [SEP] Candidate: <candidate> [SEP]
        Output format:
            Using [CLS] token as the representation of the whole sequence.

        Apply list-wise rank loss:
            1. List Net
            2. NDCG Loss
            3. ListMLE

    """

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

        Using the embeddings of the prompt 'source', 'candidate1', 'candidate2' to
        do the similarity function
            1. dot product (DP)
            2. L2 distance (L2)
            3. negative log likelihood base on softmax (NLL)
            4. cosine similarity (Cos)

        Using Contrastive Loss function
            1. NT-Xent
            2. InfoNCE from SimCLR
            3. MoCo (momentum contrastive)
            4. BYOL (bootstrap your own latent)
            5. Barlow Twins
    """
    def __init__(self, pretrained_model, args):
        super(SCR, self).__init__()
        self.args = args
        self.n_tasks = self.args["n_tasks"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.top_k_permutation = self.args.get("top_k_permutation", 1)
        self.drop_out = self.args.get("drop_out", 0.05)

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.hidden_size

        if self.loss_type == "BCE":
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

        elif self.loss_type == "ListMLE":
            self.regression_layer = nn.Sequential(
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_out),
                nn.Linear(2 * self.hidden_size, 1),
            )
        elif self.loss_type == "ListNet":
            self.regression_layer = nn.Sequential(
                nn.Dropout(self.drop_out),
                nn.Linear(self.hidden_size, 2 * self.hidden_size),
                nn.Tanh(),
                nn.Dropout(self.drop_out),
                nn.Linear(2 * self.hidden_size, 1),
            )

    def BCE_loss(self, encs, labels):
        """
            SummareReranker
        Args:
            encs: [batch_size, n_pairs, hidden_size]
            labels: [batch_size, n_pairs, n_task]
        Return:
            loss: [1]
            preds: [batch_size, n_pairs]
        """
        batch_size, n_pairs, hidden_size = encs.shape
        batch_size, n_pairs, n_task = labels.shape
        # shared bottom
        encs = self.fc2(self.relu(self.fc1(encs))).reshape(batch_size * n_pairs, -1)
        # MoE
        moe_preds, aux_loss = self.moe(encs, train = self.training, collect_gates = not(self.training))
        # go to towers for different tasks
        pred_scores = torch.cat([
            tower(moe_pred) for moe_pred, tower in zip(moe_preds, self.towers)
        ], dim=-1).reshape(batch_size, n_pairs, n_task)
        # compute loss
        loss = F.binary_cross_entropy_with_logits(pred_scores, labels, reduction="none")
        torch.sum(torch.mean(loss, dim=(1,2)))
        loss += aux_loss

        # return loss and logits
        pred_scores = torch.mean(pred_scores, dim=-1).detach().reshape(batch_size, n_pairs)
        pred_scores = self.sigmoid(pred_scores)
        return loss, pred_scores

    def forward(self, input_ids, attention_mask, scores, sep_token):
        """
        Args:
            input_ids: [batch_size, n_pairs, seq_len]
            attention_mask: [batch_size, n_pairs, seq_len]
            scores: [batch_size, n_pairs, n_task]
        """
        device = input_ids.device

        batch_size, n_pairs, seq_len = input_ids.shape
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(device)

        if self.training:
            # sub sampling candidates if needed
            if self.num_pos > 0 and self.num_neg > 0:
                sub_idx, input_ids, attention_mask, scores, labels = \
                    sub_sampling(input_ids, attention_mask, scores, labels, self.num_pos, self.num_neg)
        loss = torch.tensor(0.0).to(device)

        to_model_input_ids = input_ids.view(-1, seq_len)
        to_model_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=to_model_input_ids,
            attention_mask=to_model_attention_mask,
            output_hidden_states = True
        )

        # select the representation of the sep_token from the 2 candidates
        encs = outputs["last_hidden_state"].reshape(batch_size, n_pairs, seq_len, self.hidden_size)
        for i in range(batch_size):
            for j in range(n_pairs):
                sep_indices = torch.nonzero(torch.eq(input_ids[i,j], sep_token)).squeeze(1)
                assert len(sep_indices) == 3, "sep_token should be 3"
                sep_indices = sep_indices[:2]
                encs[i,j] = encs[i,j,sep_indices]
        encs_1 = encs[:, :, 0] # [batch_size, n_pairs, hidden_size]
        encs_2 = encs[:, :, 1] # [batch_size, n_pairs, hidden_size]


        # compute similarity matrix
        encs_1 = F.normalize(encs[:, :, 0], dim=-1)
        encs_2 = F.normalize(encs[:, :, 1], dim=-1)
        sim_mat = torch.matmul(encs_1, encs_2.transpose(1, 2)).squeeze(1) # [batch_size, n_candidate]
        sum_scores = torch.sum(scores, dim=-1).to(device) # [batch_size, n_candidate]

        # TODO: compute pairwise loss

        if self.loss_type == "BCE":
            loss = F.binary_cross_entropy_with_logits(sim_mat, labels)
        elif self.loss_type == "infoNCE":
            loss = infoNCE_loss(sim_mat, labels)
        elif self.loss_type == "ListMLE":
            loss = ListMLE_loss(sim_mat, sum_scores)
        elif self.loss_type == "ListNet":
            loss = ListNet_loss(sim_mat, sum_scores, self.top_k_permutation)
        else:
            raise ValueError("Loss type not supported")

        outputs = {
            "loss": loss,
            "preds": sim_mat,
        }
        return outputs



class T5CrossCompareReranker(nn.Module):
    """
        Seq2seq T5 Cross Compare Reranker (Cross encoder version of Dual Encoder)
        Using T5 as backbone,

        Given a source text and 2 generated candidates,
        this ranker will compare the 2 candidates and give the better one by
        doing cross attention between query and 2 candidates .

        Input format:
            [CLS] source: <source> [SEP] candidate1: <candidate1> [SEP] candiate2: <candidate2> [SEP] Is candidate1 better than candidate2 considering task <task>? [SEP]
        Output format:
            'true' or 'false'

        See paper:http://arxiv.org/abs/2003.06713 for details

    """


class T5CompareGenReranker(nn.Module):
    """
        Compare and Generation Reranker
        Using Seq2Seq T5 as backbone, 1024 input length

        Give a source text and 2 or more generated candidates,
        this ranker will compare the 2 or more candidates and give the better one by
        learning to generate the better one.

        Input format:
            [CLS] source: <source> [SEP] candidate1: <candidate1> [SEP] candiate2: <candidate2> [SEP] Please generate the better one of the 2 candiate considering task <task>: [SEP]
        Output format:
            better candidate

        Using Contrastive Loss function, not the cross entropy loss
        Only Comparein
            1. NT-Xent
            2. InfoNCE from SimCLR
            3. MoCo (momentum contrastive)
            4. BYOL (bootstrap your own latent)
            5. Barlow Twins
    """

def sub_sampling(input_ids, attention_mask, scores, labels, num_pos, num_neg, mode="bottom"):
    """
    Args:
        input_ids: [batch_size, n_candidate, seq_len]
        attention_mask: [batch_size, n_candidate, seq_len]
        scores: [batch_size, n_candidate, n_task]
        labels: [batch_size, n_candidate, n_task]
        num_pos: int
        num_neg: int
        mode: str, "bottom" or "random"
    """
    batch_size, n_candidate, seq_len = input_ids.shape
    selected_idx = []
    for i in range(batch_size):
        idx = np.arange(n_candidate)
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
        pos_idx = sorted_idx[:num_pos]
        if mode == "bottom":
            neg_idx = sorted_idx[-num_neg:]
        elif mode == "random":
            neg_idx = np.random.choice(sorted_idx[num_pos:], num_neg, replace=False)
        else:
            raise NotImplementedError
        idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(idx)
        idx = unique_idx[idx]
        selected_idx.append(idx)
    selected_idx = torch.tensor(selected_idx).to(input_ids.device)
    input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]
    return selected_idx, input_ids, attention_mask, scores, labels
