import os
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dualfid.model_moe import MoE
from dualfid.loss import (
    ListMLE_loss,
    ListNet_loss,
    infoNCE_loss
)
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
        _, n_candidates, _ = x.size()
        pred_scores = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(n_candidates):
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

class Discriminator(nn.Module):

    def __init__(self, size):
        super(Discriminator, self).__init__()
        self.size = size
        self.linear_shortcut = nn.Linear(2*self.size, 1)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(2*self.size, self.size, bias=False),
            nn.BatchNorm1d(self.size),
            nn.ReLU(),
            nn.Linear(self.size, 1)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, z):
        """
        Args:
            x: [batch_size, size]
            z: [batch_size, size]

        Returns:
            [batch_size]
        """
        batch_size = x.size(0)
        joint_xz = torch.cat([x, z], dim=-1)
        joint_outs = (
            self.linear_shortcut(joint_xz) + self.block_nonlinear(joint_xz)
        ).squeeze(-1)
        # apply JSD activation
        joint_outs = self.sigmoid(joint_outs)
        return joint_outs

class HeadLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, drop_out=0.03):
        super(HeadLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):
        return self.layer(x)


class MIDisentangledLayer(nn.Module):
    """
        Disentangled Mutual Information (MI) Layer.
    """
    def __init__(
        self,
        hidden_size,
        drop_out=0.03,
    ):
        """
        Args:
            hidden_size: the hidden size of the 2 input encodings.
            drop_out: the dropout rate.
        """
        super(MIDisentangledLayer, self).__init__()

        self.hidden_size = hidden_size
        self.drop_out = drop_out
        # layer for compute shared embedding
        self.sh_layer = HeadLayer(2 * self.hidden_size, 4 * self.hidden_size, self.hidden_size, self.drop_out)
        # layer for compute exclusive embedding
        self.ex_layer = HeadLayer(2 * self.hidden_size, 4 * self.hidden_size, self.hidden_size, self.drop_out)
        # layers for computing mutual information
        self.discriminator = Discriminator(self.hidden_size)
        self.co_reconstructor = HeadLayer(2 * self.hidden_size, 4 * self.hidden_size, self.hidden_size, self.drop_out)
        self.sh_reconstructor = HeadLayer(1 * self.hidden_size, 2 * self.hidden_size, self.hidden_size, self.drop_out)


    def forward(self, X, Y):
        """
            Get the exclusive encodings of cand1 and cand2
        Args:
            X: [batch_size, hidden_size]
            Y: [batch_size, hidden_size]
            sh_alpha: float, the weight of shared loss
            ex_alpha: float, the weight of exclusive loss
        """
        batch_size, hidden_size = X.shape # here the batch_size is pseudo batch_size
        # get the shared and exclusive encodings of cand1 and cand2

        sh_XY = self.sh_layer(torch.cat([X, Y], dim=-1))
        ex_X = self.ex_layer(torch.cat([sh_XY, X], dim=-1))
        ex_Y = self.ex_layer(torch.cat([sh_XY, Y], dim=-1))

        rand_int = random.randint(1, batch_size-1)
        batch_idx = torch.arange(batch_size)
        random_idx = torch.cat([batch_idx[rand_int:], batch_idx[:rand_int]], dim=-1)

        # compute the ex mutual information; to minimize the mutual information between ex_X, ex_Y and sh_XY
        MI_ex_X_pos = self.discriminator(ex_X, sh_XY)
        MI_ex_X_neg = self.discriminator(ex_X, sh_XY[random_idx])
        MI_ex_Y_pos = self.discriminator(ex_Y, sh_XY)
        MI_ex_Y_neg = self.discriminator(ex_Y, sh_XY[random_idx])

        MI = (MI_ex_X_pos + MI_ex_Y_pos - MI_ex_X_neg - MI_ex_Y_neg).mean() / 2.0

        # reconstruct the original sentence from shared and exclusive encodings
        recon_X = self.co_reconstructor(torch.cat([sh_XY, ex_X], dim=-1))
        recon_Y = self.co_reconstructor(torch.cat([sh_XY, ex_Y], dim=-1))
        LOSS = F.mse_loss(recon_X, X) + F.mse_loss(recon_Y, Y)
        sh_recon = self.sh_reconstructor(sh_XY)
        LOSS += F.mse_loss(sh_recon, X) + F.mse_loss(sh_recon, Y)
        LOSS /= 4.0
        LOSS += MI

        return LOSS, ex_X, ex_Y
