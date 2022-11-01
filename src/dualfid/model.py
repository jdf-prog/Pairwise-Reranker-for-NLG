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
import torch.nn.functional as F
from dualfid.wrapper import (
    FiDEncoderWrapper,
    DualFiDEncoderWrapper,
    DecoderWrapper,
)
from dualfid.model_util import (
    regression_BCE_loss,
)
from transformers.models.bart.modeling_bart import shift_tokens_right
class DualFiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs
        )

    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.model.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )

    def wrap_model(self):
        """
        Wrap the model's encoder, decoder with DualFIDEncoderWrapper, FualFIDDecoderWrapper.
        """
        encoder1 = self.model.encoder
        encoder2 = copy.deepcopy(encoder1)
        encoder2.embed_tokens = encoder1.embed_tokens # share the embedding
        self.model.encoder = DualFiDEncoderWrapper(
            encoder1,
            encoder2,
            self.model.shared.padding_idx,
            self.config.n_tasks,
            self.config.d_model,
            self.config.top_k_candidates,
            self.config.use_aux_loss
        )
        self.model.decoder = DecoderWrapper(
            self.model.decoder,
            self.encoder.get_attention_mask_for_decoder
        )
    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.model.encoder = self.model.encoder.encoder1
        self.model.decoder = self.model.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        Args:
            scores: [batch_size, n_candidates, n_tasks]
        Returns:
            pred_scores: [batch_size, n_candidates]
                the aggregated prediction score for direct selction
            loss: torch.Tensor, float loss
        """
        x, aux_loss = self.encoder.get_multi_task_layer_output()
        return regression_BCE_loss(x, aux_loss, scores)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        """
            override the prepare_inputs_for_generation method in BartForConditionalGeneration
        """
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # expand the encoder attention mask for generation
        fused_attention_mask = self.encoder.get_attention_mask_for_decoder()
        expand_size = attention_mask.size(0) // fused_attention_mask.size(0) # i.e. beam size
        original_batch_size = fused_attention_mask.size(0)
        expanded_return_idx = (
            torch.arange(original_batch_size).view(-1, 1).repeat(1, expand_size).view(-1).to(decoder_input_ids.device)
        )
        fused_attention_mask = fused_attention_mask.index_select(0, expanded_return_idx)
        attention_mask = fused_attention_mask

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def shared(self):
        return self.model.shared

class DualFiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

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
    def generate(self, input_ids, attention_mask, **kwargs):
        self.encoder.n_ctx = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            **kwargs
        )

    def wrap_model(self):
        """
        Wrap the model's encoder, decoder with DualFiDEncoderWrapper, DualFIDDecoderWrapper.
        """
        encoder1 = self.encoder
        encoder2 = copy.deepcopy(encoder1)
        encoder2.embed_tokens = encoder1.embed_tokens # share the embedding
        self.encoder = DualFiDEncoderWrapper(
            encoder1,
            encoder2,
            self.config.pad_token_id,
            self.config.n_tasks,
            self.config.d_model,
            self.config.top_k_candidates,
            self.config.use_aux_loss
        )
        self.decoder = DecoderWrapper(self.decoder, self.encoder.get_attention_mask_for_decoder)

    def unwrap_model(self):
        """
        Unwrap the model's encoder, decoder
        """
        self.encoder = self.encoder.encoder1
        self.decoder = self.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        Args:
            scores: [batch_size, n_candidates, n_tasks]
        Returns:
            pred_scores: [batch_size, n_candidates]
                the aggregated prediction score for direct selction
            loss: torch.Tensor, float loss
        """
        x, aux_loss = self.encoder.get_multi_task_layer_output()
        return regression_BCE_loss(x, aux_loss, scores)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        # expand the encoder attention mask for generation
        fused_attention_mask = self.encoder.get_attention_mask_for_decoder()
        expand_size = attention_mask.size(0) // fused_attention_mask.size(0) # i.e. beam size
        original_batch_size = fused_attention_mask.size(0)
        expanded_return_idx = (
            torch.arange(original_batch_size).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        fused_attention_mask = fused_attention_mask.index_select(0, expanded_return_idx)
        attention_mask = fused_attention_mask

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

class FiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_model()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.model.encoder.n_ctx = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
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

    def wrap_model(self):
        """
        Wrap BART encoder to obtain a Fusion-in-Decoder model.
        """
        self.model.encoder = FiDEncoderWrapper(
            self.model.encoder,
            self.config.n_tasks,
            self.config.d_model,
            self.config.top_k_candidates,
            self.config.use_aux_loss
        )
        self.model.decoder = DecoderWrapper(
            self.model.decoder,
            self.model.encoder.get_attention_mask_for_decoder)

    def unwrap_model(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load BART weights.
        """
        self.model.encoder = self.model.encoder.encoder
        self.model.decoder = self.model.decoder.decoder

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_model()
        self.load_state_dict(state_dict)
        self.wrap_model()

    def compute_auxiliary_loss(self, scores):
        """
        Compute the auxiliary loss
        Args:
            scores: [batch_size, n_candidates, n_tasks]
        Returns:
            pred_scores: [batch_size, n_candidates]
                the aggregated prediction score for direct selction
            loss: torch.Tensor, float loss
        """
        x, aux_loss = self.model.encoder.get_multi_task_layer_output()
        return regression_BCE_loss(x, aux_loss, scores)

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        """
            override the prepare_inputs_for_generation method in BartForConditionalGeneration
        """
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]


        # expand the encoder attention mask for generation
        fused_attention_mask = self.encoder.get_attention_mask_for_decoder()

        expand_size = attention_mask.size(0) // fused_attention_mask.size(0) # i.e. beam size
        original_batch_size = fused_attention_mask.size(0)
        expanded_return_idx = (
            torch.arange(original_batch_size).view(-1, 1).repeat(1, expand_size).view(-1).to(decoder_input_ids.device)
        )
        fused_attention_mask = fused_attention_mask.index_select(0, expanded_return_idx)
        attention_mask = fused_attention_mask

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

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
        self.encoder = FiDEncoderWrapper(
            self.encoder,
            self.config.n_tasks,
            self.config.d_model,
            self.config.top_k_candidates,
            self.config.use_aux_loss
        )
        self.decoder = DecoderWrapper(
            self.decoder,
            self.encoder.get_attention_mask_for_decoder)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_hfm(self, state_dict):
        """ load huggingface model """
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        # expand the encoder attention mask for generation
        fused_attention_mask = self.encoder.get_attention_mask_for_decoder()
        expand_size = attention_mask.size(0) // fused_attention_mask.size(0) # i.e. beam size
        original_batch_size = fused_attention_mask.size(0)
        expanded_return_idx = (
            torch.arange(original_batch_size).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        fused_attention_mask = fused_attention_mask.index_select(0, expanded_return_idx)
        attention_mask = fused_attention_mask

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }



