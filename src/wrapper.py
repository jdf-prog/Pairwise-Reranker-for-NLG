import torch
import torch.nn.functional as F
from torch import nn
import copy
from transformers.modeling_outputs import BaseModelOutput
from scipy.stats import pearsonr
import torch.nn.functional as F
from src.model_moe import MoE, MLPTower
import numpy as np

class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_ctx * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_ctx
        input_ids = input_ids.view(bsz*self.n_ctx, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_ctx, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].reshape(bsz, self.n_ctx*passage_length, -1), ) + outputs[1:]
        return outputs

class DualEncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Wrapper to obtain a Dual Fusion-in-Decoder model.
    """
    def __init__(self, encoder1, encoder2, padding_idx=0, n_tasks=-1, d_model=512):
        """
        Args:
            encoder1: the encoder for source
            encoder2: the encoder for candidates
            padding_idx: the padding token id
            n_tasks: the number of tasks
            d_model: the hidden size of the model
        """
        super().__init__()
        # duplicate encoder, one for the source, one for the candidates
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.padding_idx = padding_idx
        self.n_ctx = None # number of candidates + 1 (source), should be set before forward
        self.source_cls_embedding = None
        self.candidate_cls_embedding = None
        self.preds = None
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.multi_task_layer = ModelMultitaskRegression(n_tasks, 2 * d_model, d_model)

    def reduce_padding(self, input_ids, attention_mask):
        """
            remove the unnecessary padding at the tail to save memory.
        """
        padding_mask = input_ids.eq(self.padding_idx)
        unecessary_padding_mask = torch.prod(padding_mask, dim=0).bool()
        input_ids = input_ids[:, ~unecessary_padding_mask]
        attention_mask = attention_mask[:, ~unecessary_padding_mask]
        reduced_length = input_ids.size(1)
        return input_ids, attention_mask, reduced_length

    def select_topk(self, topk=3):
        """
        Select the topk candidates from the candidates
        Returns:
            best_pred_index: [bsz, topk, 1]
        """
        source_cls_embed, candidate_cls_embed = self.get_cls_embed()
        bzs, n_candidates, d_model = candidate_cls_embed.size()
        inputs = torch.cat((
            source_cls_embed.unsqueeze(1).repeat(1, n_candidates, 1),
            candidate_cls_embed
        ), dim=-1)
        # save the pred scores for loss computation
        preds= self.multi_task_layer(inputs)
        if self.preds is None:
            self.preds = preds
        else:
            self.preds = torch.cat((self.preds, preds), dim=0)

        # change the scores to rank
        ranks = torch.argsort(preds, dim=1).type(torch.float) # lower score get worse and lower rank
        assert ranks.shape == (bzs, n_candidates, self.n_tasks)
        # select the index of the one with the top khigghest average rank
        _, indices = torch.topk(torch.mean(ranks, dim=-1), k=topk, dim=-1)
        return indices


    def forward(self, input_ids=None, attention_mask=None, return_dict=None, **kwargs):
        assert self.n_ctx is not None, "n_ctx is not set"
        # total_length = n_ctx * ctx_length
        bsz, total_length = input_ids.shape
        ctx_length = total_length // self.n_ctx
        source_input_ids = input_ids[:, :ctx_length]
        source_attention_mask = attention_mask[:, :ctx_length]
        # get the corresponding inputs for source and candidates
        candidate_input_ids = input_ids[:, ctx_length:].reshape(bsz*(self.n_ctx-1), ctx_length)
        candidate_attention_mask = attention_mask[:, ctx_length:].reshape(bsz*(self.n_ctx-1), ctx_length)
        # reduce the candidate padding
        source_input_ids, source_attention_mask, source_length = self.reduce_padding(source_input_ids, source_attention_mask)
        candidate_input_ids, candidate_attention_mask, candidate_length = self.reduce_padding(candidate_input_ids, candidate_attention_mask)
        # encoder using difference encoder
        encoder1_outputs = self.encoder1(source_input_ids, source_attention_mask, **kwargs)
        encoder2_outputs = self.encoder2(candidate_input_ids, candidate_attention_mask, **kwargs)
        # save the cls embedding for this batch for MoE Loss
        if self.source_cls_embedding is None or self.candidate_cls_embedding is None:
            self.source_cls_embedding = encoder1_outputs[0][:, 0, :]
            self.candidate_cls_embedding = encoder2_outputs[0][:, ::candidate_length, :].reshape(bsz, self.n_ctx-1, -1)
        else:
            torch.cat((self.source_cls_embedding, encoder1_outputs[0][:, 0, :]), dim=0)
            torch.cat((self.candidate_cls_embedding, encoder2_outputs[0][:, ::candidate_length, :].reshape(bsz, self.n_ctx-1, -1)), dim=0)

        # concatenate the outputs of the 2 encoders
        outputs = tuple()
        indices = self.select_topk() # select the topk candidates (bsz, topk, 1)
        # 1. last_hidden_state
        encoder1_output = encoder1_outputs[0].reshape(bsz, source_length, -1)
        encoder2_output = encoder2_outputs[0].reshape(bsz, (self.n_ctx-1), candidate_length, -1)
        encoder2_output = encoder2_output[torch.arange(bsz).unsqueeze(1), indices, :, :].reshape(bsz, -1, self.d_model)
        outputs += (torch.cat([encoder1_output, encoder2_output], dim=1), )
        # 2. all hidden states
        if (len(encoder1_outputs) >= 2 and
            len(encoder2_outputs) >= 2 and
            encoder1_outputs[1] is not None and
            encoder2_outputs[1] is not None):
            hidden_states = tuple()
            for i in range(len(encoder1_outputs[1])):
                encoder1_output = encoder1_outputs[1][i].reshape(bsz, source_length, -1)
                encoder2_output = encoder2_outputs[1][i].reshape(bsz, (self.n_ctx-1) * candidate_length, -1)
                hidden_states += (torch.cat([encoder1_output, encoder2_output], dim=1), )
            outputs += (hidden_states, )
        else:
            outputs += (None, )
        # 3. all attentions
        if (len(encoder1_outputs) >= 3 and
            len(encoder2_outputs) >= 3 and
            encoder1_outputs[2] is not None and
            encoder2_outputs[2] is not None):
            attentions = tuple()
            for i in range(len(encoder1_outputs[2])):
                encoder1_output = encoder1_outputs[2][i].reshape(bsz, source_length, -1)
                encoder2_output = encoder2_outputs[2][i].reshape(bsz, (self.n_ctx-1) * candidate_length, -1)
                attentions += (torch.cat([encoder1_output, encoder2_output], dim=1), )
            outputs += (attentions, )
        else:
            outputs += (None, )

        # Wrap the outputs in a BaseModelOutput when return_dict is True
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=outputs[0],
                hidden_states=outputs[1] if len(outputs) > 1 else None,
                attentions=outputs[2] if len(outputs) > 2 else None,
            )
        else:
            return tuple(v for v in outputs if v is not None)

    def get_cls_embed(self):
        """
            Get the cls embedding of both encoder1 and encoder2 from the steps before
            set to empty once get
            Returns:
                source_cls_embedding: [bsz*accum_steps, hidden_size]
                candidate_cls_embedding: [bsz*accum_steps, n_ctx-1, hidden_size]
        """
        if self.source_cls_embedding is None or self.candidate_cls_embedding is None:
            raise ValueError("source_cls_embedding or candidate_cls_embedding is not set, please run forward first")
        result = (self.source_cls_embedding, self.candidate_cls_embedding)
        self.source_cls_embedding = None
        self.candidate_cls_embedding = None
        return result

    def get_pred_scores(self):
        if self.preds is None:
            raise ValueError("preds is not set, please run forward first")
        result = self.preds
        self.preds = None
        return result


class DualBartDecoderWrapper(torch.nn.Module):
    """
    Decoder Wrapper to assist the DualEncoderWrapper
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        **kwargs):
        """
            adjust the encoder padding mask to fit the padding reduce during the encoding
        """
        # After the reduce, no padding existing in the encoder_hidden_states
        # So the encoder padding mask is all True, i.e. all attending
        encoder_padding_mask = torch.ones_like(encoder_hidden_states[:, :, 0]).bool()
        return self.decoder(
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            **kwargs
        )

class DualT5DecoderWrapper(torch.nn.Module):
    """
    Decoder Wrapper to assist the DualEncoderWrapper
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        **kwargs):
        """
            adjust the encoder padding mask to fit the padding reduce during the encoding
        """
        # After the reduce, no padding existing in the encoder_hidden_states
        # So the encoder padding mask is all True, i.e. all attending
        encoder_attention_mask = torch.ones_like(encoder_hidden_states[:, :, 0]).bool()
        return self.decoder(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            **kwargs
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
        return x






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

    def __init__(self, device, n_tasks, input_size, hidden_size, num_experts=6, expert_hidden_size=1024, k=3, tower_hidden_size=1024):
        super(ModelMultitaskBinary, self).__init__()
        self.device = device
        self.n_tasks = n_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_hidden_size = expert_hidden_size
        self.k = k
        self.tower_hidden_size = tower_hidden_size
        # shared bottom
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # MoE
        self.moe = MoE(device, n_tasks, hidden_size, hidden_size, num_experts, expert_hidden_size, k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(hidden_size, tower_hidden_size) for i in range(n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

        # sampled candidates
        self.selected_idx = []

        # training labels
        self.original_training_labels = {}
        self.training_labels = {}
        self.training_scores = {}
        self.training_hits = {}
        for j in range(n_tasks):
            self.original_training_labels[j] = []
            self.training_labels[j] = []
            self.training_scores[j] = []
            self.training_hits[j] = []

        # multi-summary evaluation
        self.multi_summary_pred_idx = {}
        self.multi_summary_preds = {}
        for j in range(n_tasks):
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

    def forward(self, cls_embed, scores, train=True):
        bzs, n_candidate, d_input = cls_embed.shape
        d_model = d_input // 2
        loss = torch.tensor(0.0).to(cls_embed.device)
        total_predictions_idx = []
        overall_sums = []
        overall_predictions = []
        for i in range(n_candidate):
            # labels construction
            scores_i = scores[i]
            original_scores_i = scores_i.clone().detach()
            labels_i = torch.zeros(scores_i.shape, device = cls_embed.device)
            for j in range(self.n_tasks):
                best_j = scores_i[j].max()
                if self.args.sharp_pos:
                    if best_j > scores_i[j].min():
                        labels_i[j][scores_i[j] == best_j] = 1
                else:
                    labels_i[j][scores_i[j] == best_j] = 1
            original_labels_i = labels_i.clone().detach()
            # model output
            encs = encs[:, i, :] # [CLS]
            # shared bottom
            if self.args.use_shared_bottom:
                preds_i = self.fc2(self.relu(self.fc1(encs)))
            else:
                preds_i = encs
            # MoE
            preds_i, aux_loss_i = self.moe(preds_i, train = train, collect_gates = not(train))

            loss_i = torch.tensor(0.0).to(cls_embed.device)
            total_predictions = np.zeros(len(preds_i[0]))
            for j in range(self.n_tasks):

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


