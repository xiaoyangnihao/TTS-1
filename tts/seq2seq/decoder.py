"""Tacotron Decoder"""

import torch
import torch.nn as nn
from tts.layers.attention import DynamicConvolutionAttention
from tts.layers.common import PreNet


def zoneout(prev, current, p=0.1):
    """Apply zoneout
    """
    mask = torch.empty_like(prev).bernoulli_(p)

    return mask * prev + (1 - mask) * current


class Decoder(nn.Module):
    """Tacotron seq2seq decoder
    """
    def __init__(self, n_mels, memory_dim, prenet_layer_sizes, dropout,
                 attn_rnn_size, attn_dim, static_channels, static_kernel_size,
                 dynamic_channels, dynamic_kernel_size, prior_len, alpha, beta,
                 decoder_rnn_size, zoneout, r):
        """Instantiate the decoder
        """
        super().__init__()

        self.n_mels = n_mels
        self.memory_dim = memory_dim
        self.prenet_layer_sizes = prenet_layer_sizes
        self.dropout = dropout
        self.attn_rnn_size = attn_rnn_size
        self.attn_dim = attn_dim
        self.static_channels = static_channels
        self.static_kernel_size = static_kernel_size
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_len = prior_len
        self.alpha = alpha
        self.beta = beta
        self.decoder_rnn_size = decoder_rnn_size
        self.zoneout = zoneout
        self.r = r

        # Prenet
        self.prenet = PreNet(in_dim=n_mels,
                             layer_sizes=prenet_layer_sizes,
                             dropout=dropout)

        # Attention
        self.attention = DynamicConvolutionAttention(
            query_dim=attn_rnn_size,
            memory_dim=memory_dim,
            attn_dim=attn_dim,
            static_channels=static_channels,
            static_kernel_size=static_kernel_size,
            dynamic_channels=dynamic_channels,
            dynamic_kernel_size=dynamic_kernel_size,
            prior_len=prior_len,
            alpha=alpha,
            beta=beta)

        # Attention RNN
        self.attn_rnn = nn.LSTMCell(memory_dim + prenet_layer_sizes[-1],
                                    attn_rnn_size)

        # Decoder projection
        self.decoder_projection = nn.Linear(memory_dim + attn_rnn_size,
                                            decoder_rnn_size)

        # Decoder
        self.decoder_rnn1 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)
        self.decoder_rnn2 = nn.LSTMCell(decoder_rnn_size, decoder_rnn_size)

        # Output projection
        self.output_projection = nn.Linear(decoder_rnn_size,
                                           n_mels * r,
                                           bias=False)

    def forward(self, y, memory, attention_weights, attention_context,
                attn_rnn_hx, decoder_rnn1_hx, decoder_rnn2_hx):
        """Forward pass
        """
        B, N = y.size()

        # Prenet
        y = self.prenet(y)

        # Compute query for current timestep
        attn_rnn_h, attn_rnn_c = self.attn_rnn(
            torch.cat((y, attention_context), dim=-1), attn_rnn_hx)
        if self.training:
            attn_rnn_h = zoneout(attn_rnn_hx[0], attn_rnn_h, p=self.zoneout)

        # Compute attention weights and attention context for current timestep
        attention_weights = self.attention(attn_rnn_h, attention_weights)
        attention_context = torch.matmul(attention_weights.unsqueeze(1),
                                         memory).squeeze(1)

        # Decoder projection
        decoder_input = self.decoder_projection(
            torch.cat((attn_rnn_h, attention_context), dim=-1))

        # Decoder RNNs
        decoder_rnn1_h, decoder_rnn1_c = self.decoder_rnn1(
            decoder_input, decoder_rnn1_hx)
        if self.training:
            decoder_rnn1_h = zoneout(decoder_rnn1_hx[0],
                                     decoder_rnn1_h,
                                     p=self.zoneout)
        decoder_input = decoder_input + decoder_rnn1_h

        decoder_rnn2_h, decoder_rnn2_c = self.decoder_rnn2(
            decoder_input, decoder_rnn2_hx)
        if self.training:
            decoder_rnn2_h = zoneout(decoder_rnn2_hx[0],
                                     decoder_rnn2_h,
                                     p=self.zoneout)
        decoder_input = decoder_input + decoder_rnn2_h

        # Output projection
        y = self.output_projection(decoder_input).view(B, N, 2)

        return y, attention_weights, attention_context, (
            attn_rnn_h, attn_rnn_c), (decoder_rnn1_h,
                                      decoder_rnn1_c), (decoder_rnn2_h,
                                                        decoder_rnn2_c)
