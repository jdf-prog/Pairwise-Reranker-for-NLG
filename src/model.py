# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Some of the codes in this file are inherited from facebook FiD
# https://github.com/facebookresearch/FiD
# We thank the authors for their great work.

import transformers
import torch
import torch.nn as nn
import numpy as np
import copy
from src.wrapper import (
    DualEncoderWrapper,
    DualBartDecoderWrapper,
    DualT5DecoderWrapper,
)
from scipy.stats import pearsonr
from transformers.models.bart.modeling_bart import shift_tokens_right

from src.model_moe import MoE, MLPTower
from src.candidate_sampling import candidate_subsampling


class DualFiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()
        self.wrap_decoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the BART forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_ctx = input_ids.size(1) - 1
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        # generate decoder input_ids from labels instead of input_ids
        decoder_input_ids = shift_tokens_right(
            labels,
            self.config.pad_token_id,
            self.config.decoder_start_token_id)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.model.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self):
        """
        Wrap BART encoder to obtain a dual encoder for the question and passage to
        obtain a Fusion-in-Decoder model.
        """
        encoder1 = self.model.encoder
        encoder2 = copy.deepcopy(encoder1)
        self.model.encoder = DualEncoderWrapper(encoder1, encoder2, self.model.shared.padding_idx)

    def wrap_decoder(self):
        self.model.decoder = DualBartDecoderWrapper(self.model.decoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load BART weights.
        """
        self.model.encoder = self.model.encoder.encoder1

    def unwrap_decoder(self):
        self.model.decoder = self.model.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_encoder()
        self.unwrap_decoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
        self.wrap_decoder()

class DualFiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()
        self.wrap_decoder()

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_encoder(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        encoder1 = self.encoder
        encoder2 = copy.deepcopy(encoder1)
        self.encoder = DualEncoderWrapper(encoder1, encoder2, self.config.pad_token_id)

    def wrap_decoder(self):
        self.decoder = DualT5DecoderWrapper(self.decoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder1

    def unwrap_decoder(self):
        self.decoder = self.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_encoder()
        self.unwrap_decoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
        self.wrap_decoder()

class ModelMultitaskBinary(nn.Module):
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

    def __init__(self, pretrained_model, tokenizer, args):
        super(ModelMultitaskBinary, self).__init__()
        self.tokenizer = tokenizer
        self.args = args

        # LM
        self.pretrained_model = pretrained_model
        # shared bottom
        self.fc1 = nn.Linear(args.hidden_size, args.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.bottom_hidden_size, args.hidden_size)
        # MoE
        self.moe = MoE(args.device, args.n_tasks, args.hidden_size, args.hidden_size, args.num_experts, args.expert_hidden_size, args.k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(args.hidden_size, args.tower_hidden_size) for i in range(args.n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

        # sampled candidates
        self.selected_idx = []

        # training labels
        self.original_training_labels = {}
        self.training_labels = {}
        self.training_scores = {}
        self.training_hits = {}
        for j in range(self.args.n_tasks):
            self.original_training_labels[j] = []
            self.training_labels[j] = []
            self.training_scores[j] = []
            self.training_hits[j] = []

        # multi-summary evaluation
        self.multi_summary_pred_idx = {}
        self.multi_summary_preds = {}
        for j in range(self.args.n_tasks):
            self.multi_summary_pred_idx[j] = []
            self.multi_summary_preds[j] = []

    def display_selected_idx(self):
        print("\nStatistics on sampled candidates:")
        n_methods = len(self.args.generation_methods)
        selected_methods = {}
        for i in range(len(self.selected_idx)):
            idx = self.selected_idx[i]
            method = int(idx / self.args.num_beams)
            if not(method in selected_methods.keys()):
                selected_methods[method] = 0
            selected_methods[method] += 1
        for method in selected_methods.keys():
            print("Generation method {}, # selected candidates: {} ({:.4f}%)".format(
                method, selected_methods[method], 100 * selected_methods[method] / len(self.selected_idx)
            ))

    def display_training_labels(self):
        print("\nStatistics on training labels:")
        for j in range(self.args.n_tasks):
            s_ori_pos_j = np.sum(self.original_training_labels[j])
            s_pos_j = np.sum(self.training_labels[j])
            m_pos_j = 100 * np.mean(self.training_labels[j]) / (self.args.n_positives + self.args.n_negatives)
            m_label_j = np.mean(self.training_scores[j])
            m_hits_j = 100 * np.mean(self.training_hits[j])
            s_hits_j = np.sum(self.training_hits[j])
            print("Task {}, # original pos: {} / {} batches // # pos: {} / {} batches, % pos: {:.4f} // mean of training label: {:.4f} // % hitting the max: {:.4f}, count: {} / {}".format(
                j, s_ori_pos_j, len(self.training_labels[j]),  s_pos_j, len(self.training_labels[j]), m_pos_j, m_label_j, m_hits_j, s_hits_j, len(self.training_hits[j])
            ))

    def display_multi_summary_predictions(self):
        print("\nMulti-summary evaluation:")
        all_ms = []
        for j in range(self.args.n_tasks):
            self.multi_summary_pred_idx[j] = np.array(self.multi_summary_pred_idx[j])
            self.multi_summary_preds[j] = np.array(self.multi_summary_preds[j])
            m_j = np.mean(self.multi_summary_preds[j])
            all_ms.append(m_j)
            print("Task {}, prediction is {:.4f}".format(j, m_j))
        print("Mean over tasks: {:.4f}".format(np.mean(all_ms)))
        intersections = []
        correlations = []
        for j in range(self.args.n_tasks):
            for k in range(self.args.n_tasks):
                if k != j:
                    intersect = 100 * np.mean(self.multi_summary_pred_idx[j] == self.multi_summary_pred_idx[k])
                    intersections.append(intersect)
                    corr, p = pearsonr(self.multi_summary_pred_idx[j], self.multi_summary_pred_idx[k])
                    correlations.append(corr)
        m_intersection = np.mean(intersections)
        m_corr = np.mean(correlations)
        print("Mean intersection between pairs of pred idx: {:.4f}, mean Pearson correlation: {:.4f}".format(m_intersection, m_corr))

    def forward(self, mode, text_and_summaries_ids, text_and_summaries_mask, scores):
        loss = torch.tensor(0.0).to(self.pretrained_model.device)
        accuracy = [0 for j in range(self.args.n_tasks)]
        rank = [0 for j in range(self.args.n_tasks)]
        predictions_idx = [[] for j in range(self.args.n_tasks)]
        predictions = [[] for j in range(self.args.n_tasks)]
        total_predictions_idx = []
        overall_sums = []
        overall_predictions = []
        for i in range(text_and_summaries_ids.shape[0]):

            # data
            text_and_summaries_ids_i = text_and_summaries_ids[i]
            text_and_summaries_mask_i = text_and_summaries_mask[i]

            # labels construction
            scores_i = scores[i]
            original_scores_i = scores_i.clone().detach()
            labels_i = torch.zeros(scores_i.shape, device = self.pretrained_model.device)
            for j in range(self.args.n_tasks):
                best_j = scores_i[j].max()
                if self.args.sharp_pos:
                    if best_j > scores_i[j].min():
                        labels_i[j][scores_i[j] == best_j] = 1
                else:
                    labels_i[j][scores_i[j] == best_j] = 1
            original_labels_i = labels_i.clone().detach()

            # candidate sampling
            selected_idx, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i = candidate_subsampling(
                mode, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i, self.args
            )
            self.selected_idx += selected_idx

            # model output
            # LM encoding
            outputs_i = self.pretrained_model(
                input_ids = text_and_summaries_ids_i, attention_mask = text_and_summaries_mask_i, output_hidden_states = True
            )
            encs = outputs_i["last_hidden_state"]
            encs = encs[:, 0, :] # [CLS]
            # shared bottom
            if self.args.use_shared_bottom:
                preds_i = self.fc2(self.relu(self.fc1(encs)))
            else:
                preds_i = encs
            # MoE
            train = torch.sum(mode) > 0
            preds_i, aux_loss_i = self.moe(preds_i, train = train, collect_gates = not(train))

            loss_i = torch.tensor(0.0).to(self.pretrained_model.device)
            total_predictions = np.zeros(len(preds_i[0]))
            for j in range(self.args.n_tasks):

                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]

                # labels
                labels_i_j = labels_i[j]
                if torch.sum(mode) > 0: # train
                    self.original_training_labels[j].append(original_labels_i[j].sum().item())
                    self.training_labels[j].append(labels_i_j.sum().item())
                    if labels_i_j.sum() > 0:
                        self.training_scores[j].append(scores_i[j][labels_i_j == 1].mean().item())
                    self.training_hits[j].append(int(scores_i[j].max().item() == original_scores_i[j].max().item()))

                # loss
                loss_i_j = self.loss(preds_i_j, labels_i_j)
                loss_i = loss_i + loss_i_j

                # predictions
                preds_i_j = self.sigmoid(preds_i_j).detach().cpu().numpy()
                prediction_idx = np.argmax(preds_i_j)
                predictions_idx[j].append(prediction_idx)
                prediction = scores_i[j][prediction_idx].item()
                predictions[j].append(prediction)
                total_predictions += preds_i_j

                # accuracy
                pos_idx = scores_i[j].argmax().item()
                accuracy_i_j = 100 * int(scores_i[j][prediction_idx].item() == scores_i[j][pos_idx].item())
                accuracy[j] = accuracy[j] + accuracy_i_j

                # ranks
                ranks = rank_array(preds_i_j)
                all_pos_idx = [k for k in range(len(scores_i[j])) if scores_i[j][k].item() == scores_i[j][pos_idx].item()]
                rank_i_j = np.min(ranks[all_pos_idx])
                rank[j] = rank[j] + rank_i_j
            loss_i = loss_i / self.args.n_tasks
            if self.args.use_aux_loss:
                loss_i = loss_i + aux_loss_i
            loss = loss + loss_i
            total_predictions /= self.args.n_tasks
            total_prediction_idx = np.argmax(total_predictions)
            total_predictions_idx.append(total_prediction_idx)
            overall_sum = sum([scores_i[j][total_prediction_idx].item() for j in range(self.args.n_tasks)])
            overall_sums.append(overall_sum)
            overall_predictions.append(total_predictions)

        loss /= scores.shape[0]
        outputs = {
            "loss": loss,
            "loss_nce": loss,
            "total_predictions_idx": total_predictions_idx,
            "overall_predictions": overall_predictions
        }
        prediction_sum = 0
        for j in range(self.args.n_tasks):
            accuracy[j] /= scores.shape[0]
            outputs["accuracy_{}".format(self.args.scoring_methods[j])] = torch.tensor(accuracy[j]).float().to(loss.device)
            rank[j] /= scores.shape[0]
            outputs["rank_{}".format(self.args.scoring_methods[j])] = torch.tensor(rank[j]).float().to(loss.device)
            if torch.sum(mode) == 0:
                self.multi_summary_pred_idx[j] += predictions_idx[j]
                self.multi_summary_preds[j] += predictions[j]
            predictions[j] = np.mean(predictions[j])
            outputs["prediction_{}".format(self.args.scoring_methods[j])] = torch.tensor(predictions[j]).float().to(loss.device)
            prediction_sum += predictions[j]
        outputs["prediction_sum"] = torch.tensor(prediction_sum).float().to(loss.device)
        outputs["overall_sum"] = torch.tensor(np.mean(overall_sums)).float().to(loss.device)

        return outputs



