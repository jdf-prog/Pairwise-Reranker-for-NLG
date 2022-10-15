import os
import sys
from unittest import result
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

from dualfid.model_moe import MoE, MLPTower
from dualfid.layers import (
    MoERegression,
    ModelMultitaskRegression
)
import numpy as np

class FiDEncoderWrapper(torch.nn.Module):
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

class DualFiDEncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Wrapper to obtain a Dual Fusion-in-Decoder model.
    """
    def __init__(self, encoder1, encoder2, padding_idx=0, n_tasks=-1, d_model=512, top_k=3):
        """
            The 2 encoder should have the same d_model
        Args:
            encoder1: the encoder for source
            encoder2: the encoder for candidates
            padding_idx: the padding token id
            n_tasks: the number of tasks
            d_model: the hidden size of the model
            top_k: the number of candidates to select
        """
        super().__init__()
        # duplicate encoder, one for the source, one for the candidates
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.padding_idx = padding_idx
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.top_k = top_k

        # auxiliary layers
        self.multi_task_layer = ModelMultitaskRegression(n_tasks, 2 * d_model, d_model)
        # self.multi_task_layer = MoERegression(n_tasks, 2 * d_model, d_model)

        # the following are used to save the intermediate results
        self.preds = None
        self.aux_loss = None
        self.n_ctx = None # number of candidates + 1 (source), should be set before forward
        self.source_cls_embedding = None
        self.candidate_cls_embedding = None
        self.top_k_indices = None
        self.encoder_attention_masks = None
        self.encoder_attention_masks_used = True

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

    def update_topk_indices(self, topk=1):
        """
        Select the topk candidates from the candidates
        Args:
            topk: the number of candidates to select
        Returns:
            best_pred_index: [bsz, topk]
        """
        source_cls_embed, candidate_cls_embed = self.get_cls_embed()
        bzs, n_candidates, d_model = candidate_cls_embed.size()
        inputs = torch.cat((
            source_cls_embed.unsqueeze(1).repeat(1, n_candidates, 1),
            candidate_cls_embed
        ), dim=-1)
        # save the pred scores for loss computation
        preds, aux_loss = self.multi_task_layer(inputs)
        if self.preds is None:
            self.preds = preds
            self.aux_loss = aux_loss
        else:
            self.preds = torch.cat((self.preds, preds), dim=0)
            self.aux_loss += aux_loss

        # change the scores to rank
        ranks = torch.argsort(preds, dim=1).type(torch.float) # lower score get worse and lower rank
        assert ranks.shape == (bzs, n_candidates, self.n_tasks)
        # select the index of the one with the top khigghest average rank
        _, indices = torch.topk(torch.mean(ranks, dim=-1), k=topk, dim=-1)
        self.top_k_indices = indices
        assert self.top_k_indices.shape == (bzs, topk)
        return indices

    def _concantenate_shape(self, source_encoding, candidate_encoding):
        """
            Concatenate the source encoding with the candidate encoding
            Args:
                source_encoding: [bsz, source_length, hidden_size]
                candidate_encoding: [bsz*(self.n_ctx-1), candidate_length, hidden_size]
            Returns:
                encoding: [bsz, source_length + (self.n_ctx-1)*candidate_length, hidden_size]
        """
        bsz, source_length, hidden_size = source_encoding.size()
        _, candidate_length, hidden_size = candidate_encoding.size()
        indices = self.top_k_indices
        if indices is None:
            raise ValueError("top_k_indices is not set!")
        encoded_source = source_encoding.reshape(bsz, source_length, -1)
        encoded_candidates = candidate_encoding.reshape(bsz, (self.n_ctx-1), candidate_length, -1)
        # select the topk candidates
        topk_encoded_candidates = encoded_candidates[torch.arange(bsz).unsqueeze(1), indices, :, :].reshape(bsz, -1, hidden_size)
        reuslt = torch.cat([encoded_source, topk_encoded_candidates], dim=1)
        return reuslt

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
            self.source_cls_embedding = torch.cat((
                self.source_cls_embedding,
                encoder1_outputs[0][:, 0, :]
            ), dim=0)
            self.candidate_cls_embedding = torch.cat((
                self.candidate_cls_embedding,
                encoder2_outputs[0][:, ::candidate_length, :].reshape(bsz, self.n_ctx-1, -1)
            ), dim=0)

        # concatenate the outputs of the 2 encoders
        outputs = tuple()
        top_k = min(self.n_ctx-1, self.top_k)
        self.update_topk_indices(top_k) # select the topk candidates (bsz, topk, 1)

        # 1. last_hidden_state
        encoder_hidden_states = self._concantenate_shape(
            encoder1_outputs[0],
            encoder2_outputs[0]
        )
        encoder_attention_masks = self._concantenate_shape(
            source_attention_mask.reshape(bsz, source_length, 1),
            candidate_attention_mask.reshape(bsz*(self.n_ctx-1), candidate_length, 1)
        ).reshape(bsz, -1)
        assert encoder_hidden_states.shape == (bsz, source_length + top_k*candidate_length, self.d_model), \
            f"{encoder_hidden_states.shape} != {(bsz, source_length + top_k*candidate_length, self.d_model)}" # debug
        assert encoder_attention_masks.shape == (bsz, source_length + top_k*candidate_length), \
            f"{encoder_attention_masks.shape} != {(bsz, source_length + top_k*candidate_length)}" # debug
        # save for cross attention in the deocder
        if self.encoder_attention_masks_used:
            self.encoder_attention_masks = encoder_attention_masks
        else:
            self.encoder_attention_masks = torch.cat((self.encoder_attention_masks, encoder_attention_masks), dim=0)
        outputs += (encoder_hidden_states,)

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

    def get_attention_mask_for_decoder(self):
        """
            Get the attention mask for the decoder after
            fused the hidden states of encoder1 and encoder2
        """
        if self.encoder_attention_masks is None:
            raise ValueError("encoder_attention_mask is not set, please run forward first")
        self.encoder_attention_masks_used = True
        return self.encoder_attention_masks


    def get_multi_task_layer_output(self):
        """
            Get the output of the multi-task layer
            for computing auxiliary loss
        Returns:
            preds: [bsz, n_candidates, n_tasks]
            aux_loss: torch.tensor, float loss
        """
        if self.preds is None:
            raise ValueError("preds is not set, please run forward first")
        result = (self.preds, self.aux_loss)
        self.preds = None
        self.aux_loss = None
        return result

class DualFiDDecoderWrapper(torch.nn.Module):
    """
    Decoder Wrapper to assist the DualEncoderWrapper
    """
    def __init__(self, decoder, get_attention_mask):
        super().__init__()
        self.decoder = decoder
        self.get_attention_mask = get_attention_mask

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
        # TODO: problems exist here about the padding mask
        fused_encoder_attention_mask = self.get_attention_mask()
        # if training, then use fused mask
        # else if generation, then the original attention mask, processed in function
        # prepare_inputs_for_generation() from the main model
        if fused_encoder_attention_mask.size(0) == encoder_attention_mask.size(0):
            encoder_attention_mask = fused_encoder_attention_mask
        assert (
            encoder_attention_mask is not None and
            encoder_attention_mask.shape == encoder_hidden_states.shape[:2]
        ), f"{encoder_attention_mask.shape} != {encoder_hidden_states.shape[:2]}"
        return self.decoder(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            **kwargs
        )

