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
)
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
            1. multi-task regression (MSE loss)
            2. multi-task classification (BCE loss)

        See SummaReranker, Refsum, BRIO for details
    """
    def __init__(self, pretrained_model, n_tasks=3):
        super(SCR, self).__init__()
        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.hidden_size
        self.bottom_hidden_size = self.hidden_size * 2
        # shared bottom
        self.fc1 = nn.Linear(self.hidden_size, self.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.bottom_hidden_size, self.hidden_size)
        # MoE
        self.moe = MoE(n_tasks, self.hidden_size, self.hidden_size, 2*n_tasks, self.hidden_size, k=n_tasks)
        # towers - one for each task
        self.towers = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, scores):
        """
        Args:
            input_ids: [batch_size, n_candidate, seq_len]
            attention_mask: [batch_size, n_candidate, seq_len]
            scores: [batch_size, n_candidate, n_task]
        """

        loss = torch.tensor(0.0).to(input_ids.device)
        batch_size, n_candidate, seq_len = input_ids.shape
        n_task = scores.shape[-1]

        to_model_input_ids = input_ids.view(-1, seq_len)
        to_model_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=to_model_input_ids,
            attention_mask=to_model_attention_mask,
            output_hidden_states = True
        )
        encs = outputs["last_hidden_state"][:, 0, :] # [batch_size * n_candidate, hidden_size]
        # shared bottom
        encs = self.fc2(self.relu(self.fc1(encs)))
        # MoE
        moe_preds, aux_loss = self.moe(encs, train = self.training, collect_gates = not(self.training))
        # go to towers for different tasks
        pred_scores_for_n_tasks = torch.cat([
            tower(moe_pred) for moe_pred, tower in zip(moe_preds, self.towers)
        ], dim=-1) # [batch_size * n_candidate, n_task]
        # compute loss
        labels_for_n_tasks = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]) # only the best one, [batch_size, n_candidate, n_task]
        labels_for_n_tasks = labels_for_n_tasks.view(-1, n_task).float().to(input_ids.device)
        loss = self.loss(pred_scores_for_n_tasks, labels_for_n_tasks)

        # return loss and logits
        pred_scores_for_n_tasks = \
            self.sigmoid(pred_scores_for_n_tasks).reshape(batch_size, n_candidate, -1) # to output
        outputs = {
            "loss": loss + aux_loss,
            "preds": pred_scores_for_n_tasks,
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

        Using Contrastive Loss function
            1. NT-Xent
            2. InfoNCE from SimCLR
            3. MoCo (momentum contrastive)
            4. BYOL (bootstrap your own latent)
            5. Barlow Twins

        See DPR for details
    """

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



