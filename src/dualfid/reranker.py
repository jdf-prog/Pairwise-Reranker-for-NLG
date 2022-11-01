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
    triplet_simcls_loss
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
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.localize = self.args.get("localize", False)
        self.localize_ratio = self.args.get("localize_ratio", 0.5)
        self.drop_out = self.args.get("drop_out", 0.05)

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.hidden_size
        self.step = 0
        self.sigmoid = nn.Sigmoid()

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
                nn.Linear(2 * self.hidden_size, 1),
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
        encs = outputs["last_hidden_state"][:, 0, :] # [batch_size * n_candidate, hidden_size]
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
            input_ids: [batch_size, n_candidate, seq_len]
            attention_mask: [batch_size, n_candidate, seq_len]
            target_ids: [batch_size, seq_len]
            target_attention_mask: [batch_size, seq_len]
            scores: [batch_size, n_candidate, n_task]
        """
        self.step += 1
        labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(input_ids.device)
        if self.training:
            # sub sampling candidates if needed
            batch_size, n_candidate, seq_len = input_ids.shape
            if self.localize and self.step % 50 >= 25:
                with torch.no_grad():
                    pred_scores, _ = self._forawrd(input_ids.view(-1, seq_len), attention_mask.view(-1, seq_len))
                    pred_scores = pred_scores.reshape(batch_size, n_candidate, -1)
                    selected_idx = sub_sampling(
                        "top", self.num_pos, self.num_neg, self.localize_ratio, pred_scores
                    )
                    input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            sub_idx = sub_sampling(
                self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores
            )
            input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]

        # compute pred scores
        batch_size, n_candidate, seq_len = input_ids.shape
        pred_scores, aux_loss = self._forawrd(input_ids.view(-1, seq_len), attention_mask.view(-1, seq_len))
        pred_scores = pred_scores.reshape(batch_size, n_candidate, -1) # [batch_size, n_candidate, n_tasks]

        # transpose scores and labels to let the last dim be the number of candidates
        scores = scores.transpose(1, 2).reshape(-1, n_candidate)
        labels = labels.transpose(1, 2).reshape(-1, n_candidate)
        pred_scores = pred_scores.transpose(1, 2).reshape(-1, n_candidate) # [batch_size * n_tasks, n_candidate]
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
        loss += aux_loss
        # return loss and logits
        pred_scores = pred_scores.reshape(batch_size, -1, n_candidate).transpose(1, 2) # [batch_size, n_candidate, n_tasks]
        pred_scores = torch.mean(pred_scores, dim=-1).detach().reshape(batch_size, n_candidate)
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
    def __init__(self, pretrained_model, args):
        super(DualReranker, self).__init__()
        self.args = args
        self.sub_sampling_mode = self.args["sub_sampling_mode"]
        self.sub_sampling_ratio = self.args["sub_sampling_ratio"]
        self.num_pos = self.args["num_pos"]
        self.num_neg = self.args["num_neg"]
        self.loss_type = self.args["loss_type"]
        self.localize = self.args["localize"]
        self.localize_ratio = self.args["localize_ratio"]

        # LM
        self.source_encoder = pretrained_model
        # self.candidate_encoder = deepcopy(pretrained_model)
        self.candidate_encoder = pretrained_model
        self.hidden_size = self.source_encoder.config.hidden_size
        self.step = 0

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
            candidate_ids: [batch_size, n_candidate, candidate_len]
            candidate_attention_mask: [batch_size, n_candidate, candidate_len]
        Returns:
            scores: [batch_size, n_candidate]
            target_scores: [batch_size]
        """

        batch_size, n_candidate, candidate_seq_len = candidate_ids.shape
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
        )["last_hidden_state"][:, 0, :].reshape(batch_size, n_candidate, -1) # [batch_size, n_candidate, hidden_size]

        # compute Cosine Similarity
        source_encs = F.normalize(source_encs, dim=-1)
        candidate_encs = F.normalize(candidate_encs, dim=-1)
        sim_mat = torch.matmul(source_encs, candidate_encs.transpose(1, 2)).squeeze(1) # [batch_size, n_candidate]

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
            candidate_ids: [batch_size, n_candidate, seq_len]
            candidate_attention_mask: [batch_size, n_candidate, seq_len]
            scores: [batch_size, n_candidate, n_task]
        """
        self.step += 1
        labels = torch.eq(
            torch.sum(scores, dim=-1),
            torch.max(torch.sum(scores, dim=-1), dim=1, keepdim=True)[0]
        ).float().to(source_ids.device) # [batch_size, n_candidate]
        # subsampling
        if self.training:
            batch_size, n_candidate, seq_len = candidate_ids.shape
            if self.localize and self.step % 50 >= 40:
                with torch.no_grad():
                    pred_scores, _ = self._forward(
                        source_ids, source_attention_mask,
                        target_ids, target_attention_mask,
                        candidate_ids, candidate_attention_mask)
                    selected_idx = sub_sampling("top", self.num_pos, self.num_neg, self.localize_ratio, pred_scores.unsqueeze(-1))
                    input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                    labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            selected_idx = sub_sampling(self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores)
            input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
            labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]

        sim_mat, target_sim_mat = self._forward(
            source_ids, source_attention_mask,
            target_ids, target_attention_mask,
            candidate_ids, candidate_attention_mask)

        sum_scores = torch.sum(scores, dim=-1) # [batch_size, n_candidate]
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
    def __init__(self, pretrained_model, args):
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
        self.hidden_size = self.pretrained_model.config.hidden_size
        self.sep_token = self.pretrained_model.config.eos_token_id

        # self.regression_layer = nn.Sequential(
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, self.hidden_size),
        #     nn.Tanh(),
        #     nn.Dropout(self.drop_out),
        #     nn.Linear(self.hidden_size, 1),
        #     nn.Sigmoid(),
        # )

    def _forward(
        self,
        input_ids,
        attention_mask,
    ):
        """
            Compute scores for each candidate pairs
        Args:
            input_ids: [*, seq_len]
            attention_mask: [*, seq_len]
        Returns:
            pred_probs: [*]
        """
        original_shape = input_ids.shape[:-1]
        seq_len = input_ids.shape[-1]
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        candidate_encs_idx = input_ids.eq(self.sep_token)
        assert candidate_encs_idx.sum(-1).eq(3).all(), candidate_encs_idx.sum(-1)
        candidate_encs_idx = candidate_encs_idx.nonzero()[:, 1].reshape(-1, 3)
        encs = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]).unsqueeze(-1), candidate_encs_idx]
        source_enc = encs[:, 0]
        cand1_enc = encs[:, 1]
        cand2_enc = encs[:, 2]
        dif_sim_mat = F.cosine_similarity(source_enc, cand1_enc) - F.cosine_similarity(source_enc, cand2_enc)

        pred_probs = dif_sim_mat.view(*original_shape)
        return pred_probs


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
            candidate_pair_ids: [batch_size, n_candidate, n_candidate, candidate_len]
            candidate_pair_attention_mask: [batch_size, n_candidate, n_candidate, candidate_len]
            candidate_target_ids: [batch_size, n_candidate, candidate_len]
            candidate_target_attention_mask: [batch_size, n_candidate, candidate_len]
            scores: [batch_size, n_candidate, n_tasks]
            cand_target_dif_scores: [batch_size, n_candidate, n_tasks]
        """
        device = source_ids.device

        if self.training:
            # subsampling
            batch_size, n_candidate, n_tasks = scores.shape
            selected_idx = sub_sampling(self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, torch.sum(scores, -1)) # [batch_size, sub_n_candidate]
            candidate_pair_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), selected_idx, :, :]\
                [torch.arange(batch_size).unsqueeze(1), :, selected_idx, :] # [batch_size, sub_n_candidate, sub_n_candidate, candidate_len]
            candidate_pair_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), selected_idx, :, :]\
                [torch.arange(batch_size).unsqueeze(1), :, selected_idx, :] # [batch_size, sub_n_candidate, sub_n_candidate, candidate_len]
            candidate_target_ids = candidate_target_ids[torch.arange(batch_size).unsqueeze(1), selected_idx, :] # [batch_size, sub_n_candidate, candidate_len]
            candidate_target_attention_mask = candidate_target_attention_mask[torch.arange(batch_size).unsqueeze(1), selected_idx, :] # [batch_size, sub_n_candidate, candidate_len]
            scores = scores[torch.arange(batch_size).unsqueeze(1), selected_idx, :] # [batch_size, sub_n_candidate, n_tasks]
            cand_target_dif_scores = cand_target_dif_scores[torch.arange(batch_size).unsqueeze(1), selected_idx, :] # [batch_size, sub_n_candidate, n_tasks]
            batch_size, n_candidate, n_tasks = scores.shape # update batch_size, n_candidate, n_tasks


            random_idx = torch.randperm(n_candidate, device=device)
            candidate_pair_ids = candidate_pair_ids[:, random_idx.roll(1), random_idx, :] # [batch_size, n_candidate, candidate_len]
            candidate_pair_attention_mask = candidate_pair_attention_mask[:, random_idx.roll(1), random_idx, :]
            sum_scores = torch.mean(scores, dim=-1) # [batch_size, n_candidate]
            dif_scores = (sum_scores[:, random_idx.roll(1)] - sum_scores[:, random_idx]) # [batch_size, n_candidate]
            sum_scores_range = sum_scores.max(-1)[0] - sum_scores.min(-1)[0]
            dif_scores = dif_scores / sum_scores_range.unsqueeze(-1) # normalize
            dif_signs = torch.sign(dif_scores) # [batch_size, n_candidate]
            pred_dif_scores = self._forward(
                candidate_pair_ids,
                candidate_pair_attention_mask,
            )
            # # compute mse loss
            loss = F.mse_loss(pred_dif_scores, dif_scores)
            # compute margin ranking
            # margin = torch.abs(dif_scores)
            # loss = torch.max(torch.zeros_like(margin), margin - dif_signs * pred_dif_scores).mean()
            # print("margin", margin)
            # print("pred_dif_scores", pred_dif_scores)
            # print("dif_signs", dif_signs)




            # randomly sample some pairs for compare difference
            # random_idx = torch.randperm(n_candidate, device=device)
            # candidate_pair_ids = candidate_pair_ids[:, random_idx.roll(1), random_idx, :] # [batch_size, n_candidate, candidate_len]
            # candidate_pair_attention_mask = candidate_pair_attention_mask[:, random_idx.roll(1), random_idx, :]
            # sum_scores = torch.mean(scores, dim=-1) # [batch_size, n_candidate]
            # dif_scores = (sum_scores[:, random_idx.roll(1)] - sum_scores[:, random_idx]) # [batch_size, n_candidate]
            # candidate_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
            # candidate_labels = torch.where(dif_scores == 0, torch.ones_like(dif_scores) / 2, candidate_labels)

            # candidate_pred_probs = self._forward(
            #     candidate_pair_ids,
            #     candidate_pair_attention_mask,
            # )
            # loss = - (candidate_labels * torch.log(candidate_pred_probs + 1e-8) + (1 - candidate_labels) * torch.log(1 - candidate_pred_probs + 1e-8))
            # loss = torch.mean(loss)
            # print("pred_probs", candidate_pred_probs)
            # print("labels", candidate_labels)
            # print("idx", torch.stack([random_idx.roll(1), random_idx], dim=-1))
            # print("mean_scores:", torch.mean(scores, dim=-1))
            # print("dif_scores", dif_scores)

            # compute target loss
            # target_pred_probs = self._forward(
            #     candidate_target_ids,
            #     candidate_target_attention_mask,
            # )
            # dif_scores = torch.mean(cand_target_dif_scores, dim=-1) # [batch_size, n_candidate]
            # target_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
            # target_labels = torch.where(dif_scores == 0, torch.ones_like(dif_scores) / 2, target_labels)
            # target_loss = - (target_labels * torch.log(target_pred_probs + 1e-8) + (1 - target_labels) * torch.log(1 - target_pred_probs + 1e-8))
            # target_loss = torch.mean(target_loss)
            # loss += target_loss
            # print("target_pred_probs", target_pred_probs)
            # print("target_labels", target_labels)
            # print("target_dif_scores", dif_scores)

            outputs = {
                "loss": loss,
            }
        else:
            batch_size, n_candidate, _, candidate_len = candidate_pair_ids.shape
            scores = torch.mean(scores, dim=-1) # [batch_size, n_candidate]
            permu = torch.randperm(n_candidate).repeat(batch_size, 1).to(device) # [batch_size, n_candidate]
            cur_idx = permu[:, 0]
            outputs = {
                "loss": torch.tensor(0.0).to(device),
            }
            initial_idx = cur_idx
            next_idxs = []
            better_idxs = []
            for i in range(1, n_candidate):
                next_idx = permu[:, i]
                to_model_ids = candidate_pair_ids[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                to_model_attention_mask = candidate_pair_attention_mask[torch.arange(batch_size).unsqueeze(1), cur_idx.unsqueeze(1), next_idx.unsqueeze(1), :]
                to_model_ids = to_model_ids.view(batch_size, candidate_len)
                to_model_attention_mask = to_model_attention_mask.view(batch_size, candidate_len)
                pred_probs = self._forward(
                    to_model_ids, to_model_attention_mask,
                ) # [batch_size]
                # print("cur_idx", cur_idx)
                # print("next_idx", next_idx)
                # print("pred_probs", pred_probs)

                # compute accuracy
                better_idx = torch.where(pred_probs >= 0, cur_idx, next_idx)
                better_idxs.append(better_idx)
                next_idxs.append(next_idx)
                cur_idx = better_idx
            outputs['loss'] = torch.tensor(0.0).to(device)
            outputs["select_process"] = []
            outputs["select_process"].append(torch.stack([initial_idx] + better_idxs[:-1], dim=1))
            outputs["select_process"].append(torch.stack(next_idxs, dim=1))
            outputs["select_process"].append(torch.stack(better_idxs, dim=1))
            outputs["select_process"] = torch.stack(outputs["select_process"], dim=1) # [batch_size, 3, n_candidate]
            assert outputs["select_process"].shape == (batch_size, 3, n_candidate-1), outputs["select_process"].shape
        return outputs


class CompareGenReranker(nn.Module):
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
    def __init__(self, pretrained_model, args):
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

    def forward(
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if self.training:
            # sub sampling
            batch_size, n_pair, pair_len = input_ids.shape
            selected_idx = torch.stack([
                torch.randperm(n_pair)[:int(n_pair * self.sub_sampling_ratio)] for _ in range(batch_size)
            ], dim=0) # [batch_size, n_candidate * ratio]
            input_ids = input_ids[torch.arange(batch_size).unsqueeze(1), selected_idx, :]
            attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(1), selected_idx, :]
            labels = labels[torch.arange(batch_size).unsqueeze(1), selected_idx, :]
        outputs = self.pretrained_model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
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
        selected_idx: [batch_size, n_pos+n_neg] or [batch_size, n_candidate * ratio]

    """
    batch_size, n_candidate = scores.shape

    if mode == "uniform":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=True)
        step = torch.tensor(n_candidate / (n_candidate * ratio), dtype=torch.long)
        selected_idx = sorted_idx[:, ::step]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "top":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=True)
        selected_idx = sorted_idx[:, :int(n_candidate * ratio)]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "bottom":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=False)
        selected_idx = sorted_idx[:, :int(n_candidate * ratio)]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "random":
        selected_idx = torch.stack([
            torch.randperm(n_candidate)[:int(n_candidate * ratio)] for _ in range(batch_size)
        ], dim=0) # [batch_size, n_candidate * ratio]
    elif mode in ["top_bottom", "top_random", "random_bottom"]:
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
    elif mode == "importance":
        selected_idx = importance_sampling(scores, ratio)
    else:
        raise NotImplementedError

    return selected_idx

def importance_sampling(scores, ratio):
    """
    Args:
        scores: [batch_size, n_pairs, n_task]
        ratio: ratio of positive samples
    """
    pass
