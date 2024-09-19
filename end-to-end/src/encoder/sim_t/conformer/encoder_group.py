#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn

from typing import List, Optional, Tuple, Union
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class ConformerEncoderGroup(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        num_layers (int): The number of layers that compose the Encoder Group.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        feed_forward_macaron: Optional[torch.nn.Module],
        conv_module: torch.nn.Module,
        dropout_rate: float,
        num_layers: int,
        normalize_before: bool =True,
        concat_after: bool = False,
        stochastic_depth_rate: float =0.0,
    ):
        """Construct an EncoderLayer object."""
        super(ConformerEncoderGroup, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.num_layers = num_layers
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """
        s1 = None

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # -- Start of the i-th Encoder Group
        for j in range(0, self.num_layers):
            # -- -- FFN-Macaron Module: positionwise feed forward network
            if self.feed_forward_macaron is not None:
                residual = x
                if self.normalize_before:
                    x = self.norm_ff_macaron(x)
                x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                    self.feed_forward_macaron(x)
                )
                if not self.normalize_before:
                    x = self.norm_ff_macaron(x)

            # -- -- Multi-Headed Self-Attention Module
            residual = x
            if self.normalize_before:
                x = self.norm_mha(x)

            if cache is None:
                x_q = x
            else:
                assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
                x_q = x[:, -1:, :]
                residual = residual[:, -1:, :]
                mask = None if mask is None else mask[:, -1:, :]

            if j == 0:
                # -- -- Pre-MHA Module
                if pos_emb is not None:
                    x_att = self.self_attn(x_q, x, x, pos_emb, mask)
                else:
                    x_att = self.self_attn(x_q, x, x, mask)

                # -- -- getting the score matrix provided by the first layer of the encoder group
                if s1 is None:
                    if cache is None:
                        s1 = self.self_attn.attn # (#batch, head, time1, time1)
                    else:
                        s1 = self.self_attn.attn[:, :, -1:, -1:] # (#batch, head, 1, 1)
            else:
                # -- --- Post-MHA Module
                x_att = self.post_mha(x, s1)

            if self.concat_after:
                x_concat = torch.cat((x, x_att), dim=-1)
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = residual + stoch_layer_coeff * self.dropout(x_att)
            if not self.normalize_before:
                x = self.norm_mha(x)

            # -- Convolution Module
            if self.conv_module is not None:
                residual = x
                if self.normalize_before:
                    x = self.norm_conv(x)
                x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
                if not self.normalize_before:
                    x = self.norm_conv(x)

            # -- FFN Module
            residual = x
            if self.normalize_before:
                x = self.norm_ff(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward(x)
            )
            if not self.normalize_before:
                x = self.norm_ff(x)

            if self.conv_module is not None:
                x = self.norm_final(x)

            if cache is not None:
                x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask

    def post_mha(self, values, scores):
        """Apply the Post-MHA module based on https://arxiv.org/pdf/2304.04991.pdf
        Args:
            values (torch.Tensor): Value tensor (#batch, time2, size).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        # -- transform value
        n_batch = values.size(0)
        value = self.self_attn.linear_v(values).view(n_batch, -1, self.self_attn.h, self.self_attn.d_k)
        value = value.transpose(1, 2) # (batch, head, time2, d_k)

        # -- applying the attention context vector computed by the first layer of the encoder group
        p_attn = self.self_attn.dropout(scores)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.self_attn.h * self.self_attn.d_k)
        )  # (batch, time1, d_model)

        return x  # (batch, time1, d_model)
