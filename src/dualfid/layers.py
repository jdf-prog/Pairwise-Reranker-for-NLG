import os
import sys
import copy
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

class MILayer(nn.Module):

    def __init__(self, size, n_units=1):
        super(MILayer, self).__init__()
        self.size = size
        self.n_units = n_units
        assert size % n_units == 0
        self.size_per_unit = size // n_units
        self.linear_shortcut = nn.Linear(2*self.size_per_unit, 1)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(2*self.size_per_unit, self.size_per_unit, bias=False),
            nn.BatchNorm1d(self.size_per_unit),
            nn.ReLU(),
            nn.Linear(self.size_per_unit, n_units)
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
        x = x.view(batch_size, self.size_per_unit, -1)
        z = z.view(batch_size, self.size_per_unit, -1)
        joint_xz = torch.cat([x, z], dim=-1) # [batch_size, size_per_unit, 2*size_per_unit]
        joint_outs = (
            self.linear_shortcut(joint_xz) + self.block_nonlinear(joint_xz)
        ).squeeze(1)
        return self.sigmoid(joint_outs)

class Discriminator(nn.Module):
    """
        Discriminator for the MIL training.
    """
    def __init__(self, size):
        self.layer = nn.Sequential(
            nn.Linear(2*size, size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(size, 1),
            nn.Sigmoid()
        )

    def forward(self, sh_enc, ex_enc):
        enc = torch.cat([sh_enc, ex_enc], dim=-1)
        return self.layer(enc)

class HeadLayer(nn.Module):
    def __init__(self, size, drop_out=0.03):
        self.size = size
        self.drop_out = drop_out
        self.layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.size, 2 * self.size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.size, 1 * self.size),
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
        n_units=8,
        sh_alpha=0.5,
        sh_beta=0.5,
        ex_alpha=0.5,
        ex_beta=0.5,
        sh_lambda=0.5,
        ex_lambda=0.5,
    ):
        """
        Args:
            hidden_size: the hidden size of the 2 input encodings.
            drop_out: the dropout rate.
            n_units: the number of units in for local MI computation.
            sh_alpha: the weight for the shared global MI
            sh_beta: the weight for the shared local MI
            ex_alpha: the weight for the exclusive global MI
            ex_beta: the weight for the exclusive local MI
            sh_lambda: the weight for the shared constraint term
            ex_lambda: the weight for the exclusive constraint term
        """

        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.n_units = n_units
        self.sh_alpha = sh_alpha
        self.sh_beta = sh_beta
        self.ex_alpha = ex_alpha
        self.ex_beta = ex_beta
        # layer for compute shared embedding
        self.sh_layer = HeadLayer(self.hidden_size)
        # layer for compute exclusive embedding
        self.ex_layer = HeadLayer(self.hidden_size)
        # layer of discriminator for adversial loss
        self.discriminator = Discriminator(self.hidden_size)
        # layers for computing mutual information
        self.g_MI_sh = MILayer(self.hidden_size)
        self.g_MI_ex = MILayer(self.hidden_size)
        self.l_MI_sh = MILayer(self.hidden_size, n_units=self.n_units)
        self.l_MI_ex = MILayer(self.hidden_size, n_units=self.n_units)

    def forward(self, X, Y):
        """
            Get the exclusive encodings of cand1 and cand2
        Args:
            X: [batch_size, hidden_size]
            Y: [batch_size, hidden_size]
            sh_alpha: float, the weight of shared loss
            ex_alpha: float, the weight of exclusive loss
        """
        batch_size, hidden_size = X.shapp # here the batch_size is pseudo batch_size
        # get the shared and exclusive encodings of cand1 and cand2
        sh_X = self.sh_layer(X)
        sh_Y = self.sh_layer(Y)
        ex_X = self.ex_layer(X)
        ex_Y = self.ex_layer(Y)

        # the random idx to create pairs from different samples
        rand_int = torch.randint(1, batch_size, (batch_size,))
        random_idx = torch.cat([
            torch.arange(batch_size)[rand_int:],
            torch.arange(batch_size)[:rand_int]
        ])

        # compute shared cross local and golbal MI; to maximize
        MI_G_sh_XY_pos = self.g_MI_sh(X, sh_Y)
        MI_G_sh_YX_pos = self.g_MI_sh(Y, sh_X)
        MI_L_sh_XY_pos = self.l_MI_sh(X, sh_Y)
        MI_L_sh_YX_pos = self.l_MI_sh(Y, sh_X)
        MI_G_sh_XY_neg = self.g_MI_sh(X, sh_Y[random_idx])
        MI_G_sh_YX_neg = self.g_MI_sh(Y, sh_X[random_idx])
        MI_L_sh_XY_neg = self.l_MI_sh(X, sh_Y[random_idx])
        MI_L_sh_YX_neg = self.l_MI_sh(Y, sh_X[random_idx])
        G_SH_MI = self.sh_alpha * (MI_G_sh_XY_pos + MI_G_sh_YX_pos - MI_G_sh_XY_neg - MI_G_sh_YX_neg).mean()
        L_SH_MI = self.sh_beta * (MI_L_sh_XY_pos + MI_L_sh_YX_pos - MI_L_sh_XY_neg - MI_L_sh_YX_neg).mean()

        # compute exclusive cross local and golbal MI; to maximize
        MI_G_ex_XY_pos = self.g_MI_ex(X, ex_Y)
        MI_G_ex_YX_pos = self.g_MI_ex(Y, ex_X)
        MI_L_ex_XY_pos = self.l_MI_ex(X, ex_Y)
        MI_L_ex_YX_pos = self.l_MI_ex(Y, ex_X)
        MI_G_ex_XY_neg = self.g_MI_ex(X, ex_Y[random_idx])
        MI_G_ex_YX_neg = self.g_MI_ex(Y, ex_X[random_idx])
        MI_L_ex_XY_neg = self.l_MI_ex(X, ex_Y[random_idx])
        MI_L_ex_YX_neg = self.l_MI_ex(Y, ex_X[random_idx])
        G_EX_MI = self.ex_alpha * (MI_G_ex_XY_pos + MI_G_ex_YX_pos - MI_G_ex_XY_neg - MI_G_ex_YX_neg).mean()
        L_EX_MI = self.ex_beta * (MI_L_ex_XY_pos + MI_L_ex_YX_pos - MI_L_ex_XY_neg - MI_L_ex_YX_neg).mean()

        # same shared loss; to minimize
        SH_TERM = (sh_X - sh_Y).mean()

        # compute adversial loss TODO: check this
        scores_X_pos = self.discriminator(sh_X, ex_X)
        scores_X_neg = self.discriminator(sh_X, ex_X[random_idx])
        scores_Y_pos = self.discriminator(sh_Y, ex_Y)
        scores_Y_neg = self.discriminator(sh_Y, ex_Y[random_idx])

        scores_X_pos = -torch.log(1 - scores_X_pos)
        scores_X_neg = -torch.log(scores_X_neg)
        scores_Y_pos = -torch.log(1 - scores_Y_pos)
        scores_Y_neg = -torch.log(scores_Y_neg)

        EX_TERM = (scores_X_pos + scores_X_neg + scores_Y_pos + scores_Y_neg).mean()

        LOSS = SH_TERM + EX_TERM - (G_SH_MI + L_SH_MI + G_EX_MI + L_EX_MI)
        return LOSS, ex_X, ex_Y
