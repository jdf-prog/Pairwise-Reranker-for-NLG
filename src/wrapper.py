import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import copy
from transformers.modeling_outputs import BaseModelOutput
from scipy.stats import pearsonr
from src.model_moe import MoE, MLPTower


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
    def __init__(self, encoder1, encoder2, padding_idx=0):
        """
        Args:
            encoder1: the encoder for source
            encoder2: the encoder for candidates
            padding_idx: the padding token id
        """
        super().__init__()
        # duplicate encoder, one for the source, one for the candidates
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.padding_idx = padding_idx
        self.n_ctx = None # number of candidates + 1 (source), should be set before forward
        self.source_cls_embedding = None
        self.candidate_cls_embedding = None

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
        # 1. last_hidden_state
        encoder1_output = encoder1_outputs[0].reshape(bsz, source_length, -1)
        encoder2_output = encoder2_outputs[0].reshape(bsz, (self.n_ctx-1) * candidate_length, -1)
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

    def __init__(self, n_tasks, input_size, hidden_size, num_experts=6, expert_hidden_size=1024, k=4, tower_hidden_size=1024):
        super(MoERegression, self).__init__()
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
        self.moe = MoE(n_tasks, hidden_size, hidden_size, num_experts, expert_hidden_size, k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(hidden_size) for _ in range(n_tasks)])
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
