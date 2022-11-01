import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1
DEFAULT_EPS = 1e-10


def permutation_prob(scores, level=1):
    """
    Args:
        scores: [batch_size, n_candidate]
        level: level of the permutation probs to compute
            when level is positive, we compute the top-pos permutation probs
            when level is negative, we compute the all permutation probs (same as top-n_candidate)
            when level is 0, we compute the top-1 permutation probs (same as top-1)
    Returns:
        prob: [batch_size, A(3,level)]
            represent the probability of each permutation.
            e.g. for input three scores [0.1, 0.2, 0.3], the original permutation is 0,1,2
            For the full level computation, the 2nd dim of probs is A(3,3)=6
            each representing probs of permutation
            0,1,2, 0,2,1, 1,0,2, 1,2,0, 2,0,1, 2,1,0
    """
    probs = []
    batch_size, n_candidate = scores.size()
    cur_probs = scores / scores.sum(dim=1, keepdim=True)
    if level <= -1 or level >= n_candidate:
        level = n_candidate
    if level > 1:
        for i in range(n_candidate):
            cur_prob = cur_probs[:, i].unsqueeze(1)
            scores_except_i = torch.cat([scores[:, :i], scores[:, i+1:]], dim=1)
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
    exp_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidate]
    # compute prob
    logits = permutation_prob(exp_pred_scores, top_k_permutation)
    labels = permutation_prob(exp_scores, top_k_permutation)
    # compute cross entropy loss
    loss = torch.mean(torch.sum(-labels * torch.log(logits + 1e-10), dim=1))
    return loss

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

def p_ListMLE_loss(pred_scores, scores):
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
        alpha = torch.tensor(2**(n_candidate - i) - 1, dtype=torch.float32).to(pred_scores.device)
        probs.append(alpha*(denom_prob + numer_prob)) # [batch_size]
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
    pos_sim = sim_mat * labels / temperature
    neg_sim = sim_mat * (1 - labels) / temperature
    max_sim = torch.max(pos_sim+neg_sim, dim=1, keepdim=True)[0]
    pos_sim = torch.exp(pos_sim - max_sim)
    neg_sim = torch.exp(neg_sim - max_sim)
    pos_sim_sum = torch.sum(torch.exp(pos_sim ), dim=1)
    loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
    return loss

def triplet_loss(sim_mat, target_sim, sum_scores):
    """
    Args:
        sim_mat: [batch_size, n_candidate]
        target_sim: [batch_size]
        sum_scores: [batch_size, n_candidate]
    Return:
        loss: [1]
    """
    batch_size, n_candidate = sim_mat.shape
    sum_scores = sum_scores
    loss = torch.tensor(0.0).to(sim_mat.device)
    sorted_idx = torch.argsort(sum_scores, dim=1, descending=True) # [batch_size, n_candidate]
    for i in range(batch_size):

        sim_mat_i = sim_mat[tor]


def triplet_loss_v2(pred_scores, scores, temperature=1.0):
    """
    Args:
        Note here greater is better
        pred_scores: [batch_size, n_candidate]
        scores: [batch_size, n_candidate]
    Return:
        loss: [1]
    """
    scores = F.softmax(scores/temperature, dim=1)
    batch_size, n_candidate = pred_scores.shape
    # compute loss
    sorted_indices = torch.argsort(scores, dim=1, descending=True) # [batch_size, n_candidate]
    top_idx = sorted_indices[:, 0] # [batch_size]
    top_scores = scores[torch.arange(batch_size), top_idx] # [batch_size]
    top_pred_scores = pred_scores[torch.arange(batch_size), top_idx] # [batch_size]
    loss = torch.tensor(0.0).to(pred_scores.device)
    for i in range(1, n_candidate):
        margin = top_scores - scores[torch.arange(batch_size), sorted_indices[:, i]]
        loss += torch.mean(
            torch.max(
                torch.tensor(0.0).to(pred_scores.device),
                pred_scores[torch.arange(batch_size), sorted_indices[:, i]] - top_pred_scores + margin
            ))
    return loss

def triplet_simcls_loss(sim_mat, target_sim, sum_scores):
    """
    Args:
        sim_mat: [batch_size, n_candidate]
        target_sim: [batch_size]
        sum_scores: [batch_size, n_candidate]
    Return:
        loss: [1]
    """
    loss_func = nn.MarginRankingLoss(margin=0.0)
    loss = torch.tensor(0.0).to(sim_mat.device)
    gold_margin_loss = loss_func(target_sim.repeat(sim_mat.shape[1], 1).transpose(0, 1), sim_mat, torch.ones_like(sim_mat))
    loss += gold_margin_loss
    batch_size, n_candidate = sim_mat.shape
    sorted_idx = torch.argsort(sum_scores, dim=1, descending=True) # [batch_size, n_candidate]
    for i in range(n_candidate):
        for j in range(i+1, n_candidate):
            sim_mat_i = sim_mat[torch.arange(batch_size), sorted_idx[:, i]]
            sim_mat_j = sim_mat[torch.arange(batch_size), sorted_idx[:, j]]
            loss_func = nn.MarginRankingLoss(margin=(j - i) / n_candidate)
            margin_loss = loss_func(sim_mat_i, sim_mat_j, torch.ones_like(sim_mat_i))
            loss += margin_loss
    return loss
