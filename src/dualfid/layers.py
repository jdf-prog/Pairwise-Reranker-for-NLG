import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dualfid.model_moe import MoE

class ModelMultitaskRegression(nn.Module):
    """
        This class is used to train the model for the multitask regression task.
        Use as a layer return the loss
    """
    def __init__(self, n_tasks, input_size, hidden_size):
        super(ModelMultitaskRegression, self).__init__()
        self.n_tasks = n_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_tasks)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.sigmoid(x) # do regression on [0, 1] scale
        return x, None # no loss


class MoERegression(nn.Module):
    """
        This class is modified from the original implementation of the paper:
        SummaReranker: A Multi-Task Mixture-of-Experts Re-ranking Framework for Abstractive Summarization
        paper: https://arxiv.org/abs/2203.06569
        code: https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/summareranker/model.py
        We thank the authors for sharing their code.

        In our implementation, we get passed in embedding from dual encoder and
        apply the multitask binary classification head on top of it.
        We only this layer to compute the auxiliary loss to help the generation.
        We don't use this layer for any prediction.
    """

    def __init__(self, n_tasks, input_size, hidden_size, num_experts=None, expert_hidden_size=1024, k=None):
        super(MoERegression, self).__init__()
        self.n_tasks = n_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        if num_experts is None:
            num_experts = 2 * n_tasks
            self.num_experts = num_experts
        if k is None:
            k = num_experts // 2
            self.k = k
        # shared bottom
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # MoE
        self.moe = MoE(n_tasks, hidden_size, hidden_size, num_experts, expert_hidden_size, k)
        # towers - one for each task
        self.towers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_tasks)])
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        _, n_candidate, _ = x.size()
        pred_scores = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(n_candidate):
            encs = x[:, i, :] # [CLS]
            preds_i = self.fc2(self.relu(self.fc1(encs))) # shared bottom
            train = self.training
            preds_i, aux_loss = self.moe(preds_i, train = train, collect_gates = not(train))
            pred_scores_i = []
            for j in range(self.n_tasks):
                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]
                pred_scors_i_j = self.sigmoid(preds_i_j)
                pred_scores_i.append(pred_scors_i_j)
            pred_scores_i = torch.stack(pred_scores_i, dim=1)
            pred_scores.append(pred_scores_i)
            total_aux_loss += aux_loss
        pred_scores = torch.stack(pred_scores, dim=1)
        return pred_scores, total_aux_loss

def permutation_prob(scores, level=1):
    """
    Args:
        scores: [batch_size, n_candidate]
        level: level of the permutation probs to compute
            when level is 1, we compute the top-1 permutation probs
    Returns:
        prob: [batch_size, (n_candidate-1)*(n_candidate-2)*...(n_candidate-level)]
    """
    probs = []
    batch_size, n_candidate = scores.size()
    cur_probs = scores / scores.sum(dim=1, keepdim=True)
    if level > 1:
        for i in range(len(n_candidate)):
            cur_prob = cur_probs[:, i] # [batch_size]
            scores_except_i = torch.cat([scores[:-1, :i], scores[:-1, i+1:]], dim=1)
            next_prob = permutation_prob(scores_except_i, level=level-1) # [batch_size, (n_candidate-1)*(n_candidate-2)*...(n_candidate-level)]
            probs.append(cur_prob * next_prob)
        probs = torch.cat(probs, dim=1)
        return probs
    else:
        return cur_probs



def ListNet_loss(pred_scores, scores, top_k_permutation=1):
    """
    Args:
        pred_scores: [batch_size, n_candidate]
        scores: [batch_size, n_candidate]
        top_k_permutation: int, top k permutation to compute the loss
    Return:
        loss: [1]
        preds: [batch_size, n_candidate]
    """
    # apply exp
    exp_pred_scores = torch.exp(pred_scores - torch.max(pred_scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidate]
    exp_sum_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidate]
    # compute prob
    logits = permutation_prob(exp_pred_scores, top_k_permutation)
    labels = permutation_prob(exp_sum_scores, top_k_permutation)
    # compute cross entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss, pred_scores

def ListMLE_loss(pred_scores, scores):
    """
    Args:
        pred_scores: [batch_size, n_candidate]
        scores: [batch_size, n_candidate]
    Return:
        loss: [1]
    """
    batch_size, n_candidate = pred_scores.shape
    # apply exp
    exp_pred_scores = torch.exp(pred_scores - torch.max(pred_scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidate]
    exp_sum_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidate]

    sorted_indices = torch.argsort(exp_sum_scores, dim=1, descending=True) # [batch_size, n_candidate]
    probs = []
    for i in range(n_candidate):
        order_i_indices = sorted_indices[:, i] # [batch_size]
        left_indices = sorted_indices[:,i:] # [batch_size, n_candidate - i]
        denom_prob = -torch.log(exp_pred_scores[torch.arange(batch_size), order_i_indices])
        numer_prob = torch.log(torch.sum(exp_pred_scores[torch.arange(batch_size).unsqueeze(1), left_indices], dim=1))
        probs.append(denom_prob + numer_prob) # [batch_size]
    loss = torch.sum(torch.stack(probs, dim=1), dim=1) # [batch_size]
    loss = torch.mean(loss)
    return loss

def infoNCE_loss(sim_mat, labels, temperature=0.07):
    """
        InfoNCE loss
        See paper: https://arxiv.org/abs/2002.05709
    Args:
        sim_mat: [batch_size, n_candidate]
        labels: [batch_size, n_candidate]
        temperature: float
    Return:
        loss: [1]
    """
    # compute info loss
    pos_sim = torch.sum(sim_mat * labels, dim=-1)
    pos_sim = torch.exp(pos_sim / temperature)
    neg_sim = torch.sum(sim_mat * (1 - labels), dim=-1)
    neg_sim = torch.exp(neg_sim / temperature)
    loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
    return loss
