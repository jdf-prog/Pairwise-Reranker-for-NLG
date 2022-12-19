import os
import sys
from unittest import result
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F

from reranker.layers import (
    MoERegression,
    ModelMultitaskRegression
)
import numpy as np
from pytorch_metric_learning import distances
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)

class FiDEncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(
        self,
        encoder,
        n_tasks=-1,
        d_model=512,
        top_k_candidates=-1,
        use_aux_loss=False
    ):
        super().__init__()
        self.encoder = encoder
        self.use_aux_loss = use_aux_loss
        self.n_tasks = n_tasks
        self.top_k_candidates = top_k_candidates

        # auxiliary layer
        if self.use_aux_loss:
            self.auxiliary_layer = MoERegression(
                n_tasks=n_tasks,
                input_size=d_model,
                hidden_size=d_model,)

        # # the following are used to save the intermediate results
        self.n_ctx = None # number of fused pairs of source encoder and candidates
        self.preds = None
        self.aux_loss = None
        self.encoder_attention_mask = None
        self.encoder_attention_mask_used = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        **kwargs,):
        # total_length = n_ctx * fuse_length
        bsz, total_length = input_ids.shape
        fuse_length = total_length // self.n_ctx
        input_ids = input_ids.view(bsz*self.n_ctx, fuse_length)
        attention_mask = attention_mask.view(bsz*self.n_ctx, fuse_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs) # [bsz*n_ctx, fuse_length, d_model]

        if self.use_aux_loss:
            fuse_embeddings = outputs[0].reshape(bsz, self.n_ctx, fuse_length, -1)[:,:,0,:].detach() # debug, detach
            # save the predictions and aux_loss
            preds, aux_loss = self.auxiliary_layer(fuse_embeddings)
            if self.preds is None:
                self.preds = preds
            else:
                self.preds = torch.cat((self.preds, preds), dim=1)
            if self.aux_loss is None:
                self.aux_loss = aux_loss
            else:
                self.aux_loss += aux_loss

        if self.top_k_candidates is not None and self.top_k_candidates > 0:
            top_k_indices = torch.topk(torch.sum(preds, dim=2), self.top_k_candidates, dim=1)[1] # [bsz, top_k_candidates]

            last_hidden_states = outputs[0].reshape(bsz, self.n_ctx, fuse_length, -1)
            last_hidden_states = last_hidden_states[torch.arange(bsz).unsqueeze(1), top_k_indices] # [bsz, top_k_candidates, fuse_length, d_model]
            last_hidden_states = last_hidden_states.reshape(bsz, self.top_k_candidates*fuse_length, -1)
            attention_mask = attention_mask.reshape(bsz, self.n_ctx, fuse_length)
            attention_mask = attention_mask[torch.arange(bsz).unsqueeze(1), top_k_indices] # [bsz, top_k_candidates, fuse_length]
            attention_mask = attention_mask.reshape(bsz, self.top_k_candidates*fuse_length)
            if self.encoder_attention_mask_used:
                self.encoder_attention_mask = attention_mask
                self.encoder_attention_mask_used = False
            else:
                self.encoder_attention_mask = torch.cat((self.encoder_attention_mask, attention_mask), dim=0)
        else:
            last_hidden_states = outputs[0].reshape(bsz, self.n_ctx*fuse_length, -1)
            attention_mask = attention_mask.reshape(bsz, self.n_ctx*fuse_length)
            if self.encoder_attention_mask_used:
                self.encoder_attention_mask = attention_mask
                self.encoder_attention_mask_used = False
            else:
                self.encoder_attention_mask = torch.cat((self.encoder_attention_mask, attention_mask), dim=0)


        outputs = (last_hidden_states, ) + outputs[1:]
        # Wrap the outputs in a BaseModelOutput when return_dict is True
        if return_dict:
            return BaseModelOutput(
                last_hidden_state=outputs[0],
                hidden_states=outputs[1] if len(outputs) > 1 else None,
                attentions=outputs[2] if len(outputs) > 2 else None,
            )
        else:
            return tuple(v for v in outputs if v is not None)

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

    def get_attention_mask_for_decoder(self):
        """
            Get the attention mask for the decoder after
            fused the hidden states of encoder1 and encoder2
        """
        if self.encoder_attention_mask is None:
            raise ValueError("encoder_attention_mask is not set, please run forward first")
        self.encoder_attention_mask_used = True
        return self.encoder_attention_mask

class DualFiDEncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Wrapper to obtain a Dual Fusion-in-Decoder model.
    """
    def __init__(
        self,
        encoder1,
        encoder2,
        padding_idx=0,
        n_tasks=-1,
        d_model=512,
        top_k_candidates=-1,
        use_aux_loss=False):
        """
            The 2 encoder should have the same d_model
        Args:
            encoder1: the encoder for source
            encoder2: the encoder for candidates
            padding_idx: the padding token id
            n_tasks: the number of tasks
            d_model: the hidden size of the model
            top_k_candidates: the number of candidates to select
        """
        super().__init__()
        # duplicate encoder, one for the source, one for the candidates
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.padding_idx = padding_idx
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.top_k = top_k_candidates
        self.use_aux_loss = use_aux_loss


        # auxiliary layers
        if self.use_aux_loss:
            # self.multi_task_layer = ModelMultitaskRegression(n_tasks, 2 * d_model, d_model)
            self.multi_task_layer = MoERegression(n_tasks, 2 * d_model, d_model)
            # self.dist = distances.CosineSimilarity()

        # the following are used to save the intermediate results
        self.preds = None
        self.aux_loss = None
        self.n_ctx = None # number of candidates + 1 (source), should be set before forward
        self.source_cls_embedding = None
        self.candidate_cls_embedding = None
        self.top_k_indices = None
        self.encoder_attention_mask = None
        self.encoder_attention_mask_used = True

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

    def update_topk_indices(self):
        """
        Select the topk candidates from the candidates
        Args:
            topk: the number of candidates to select
        Returns:
            best_pred_index: [bsz, topk]
        """
        if not self.use_aux_loss:
            self.top_k_indices = None
            return None
        source_cls_embed, candidate_cls_embed = self.get_cls_embed()
        bzs, n_candidates, d_model = candidate_cls_embed.size()

        # multi-task regression
        inputs = torch.cat((
            source_cls_embed.unsqueeze(1).repeat(1, n_candidates, 1),
            candidate_cls_embed
        ), dim=-1).detach() # debug
        preds, aux_loss = self.multi_task_layer(inputs)
        # save the pred scores for loss computation
        if self.preds is None:
            self.preds = preds
            self.aux_loss = aux_loss
        else:
            self.preds = torch.cat((self.preds, preds), dim=0)
            self.aux_loss += aux_loss

        # # change the scores to rank
        ranks = torch.argsort(preds, dim=1).type(torch.float) # lower score get worse and lower rank
        assert ranks.shape == (bzs, n_candidates, self.n_tasks)
        # select the index of the one with the top khigghest average rank
        top_k = min(self.top_k, self.n_ctx - 1) if self.top_k > 0 else self.n_ctx - 1
        _, indices = torch.topk(torch.mean(ranks, dim=-1), k=top_k, dim=-1)

        self.top_k_indices = indices

        assert self.top_k_indices.shape == (bzs, top_k)
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
        encoded_source = source_encoding.reshape(bsz, source_length, -1)
        encoded_candidates = candidate_encoding.reshape(bsz, (self.n_ctx-1), candidate_length, -1)
        # select the topk candidates
        if indices is not None:
            topk_encoded_candidates = encoded_candidates[torch.arange(bsz).unsqueeze(1), indices, :, :].reshape(bsz, -1, hidden_size)
        else:
            # not select because of no auxliary loss computed
            topk_encoded_candidates = encoded_candidates.reshape(bsz, -1, hidden_size)
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
        self.update_topk_indices() # select the topk candidates (bsz, topk, 1)

        # 1. last_hidden_state
        encoder_hidden_states = self._concantenate_shape(
            encoder1_outputs[0],
            encoder2_outputs[0]
        )
        encoder_attention_mask = self._concantenate_shape(
            source_attention_mask.reshape(bsz, source_length, 1),
            candidate_attention_mask.reshape(bsz*(self.n_ctx-1), candidate_length, 1)
        ).reshape(bsz, -1)
        # save for cross attention in the deocder
        if self.encoder_attention_mask_used:
            self.encoder_attention_mask = encoder_attention_mask
        else:
            self.encoder_attention_mask = torch.cat((self.encoder_attention_mask, encoder_attention_mask), dim=0)
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
        if self.encoder_attention_mask is None:
            raise ValueError("encoder_attention_mask is not set, please run forward first")
        self.encoder_attention_mask_used = True
        return self.encoder_attention_mask


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

class DecoderWrapper(torch.nn.Module):
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
