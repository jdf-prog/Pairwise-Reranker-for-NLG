import torch
import torch.nn.functional as F
from torch import nn
import copy



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

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
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
        self.source_cls_embedding = encoder1_outputs[0][:, 0, :]
        self.candidate_cls_embedding = encoder2_outputs[0][:, ::candidate_length, :].reshape(bsz, self.n_ctx-1, -1)
        # concatenate the 2 outputs
        outputs = tuple()
        # encoder outputs
        encoder1_output = encoder1_outputs[0].reshape(bsz, source_length, -1)
        encoder2_output = encoder2_outputs[0].reshape(bsz, (self.n_ctx-1) * candidate_length, -1)
        outputs += (torch.cat([encoder1_output, encoder2_output], dim=1), )
        # hidden states and attentions
        for i in range(1, len(encoder1_outputs)):
            outputs += ((encoder1_outputs[i], encoder2_outputs[i]), )
        return outputs

    def get_cls_embedding(self):
        """
            Get the cls embedding of both encoder1 and encoder2
            during this batch step
        """
        return self.source_cls_embedding, self.candidate_cls_embedding

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
