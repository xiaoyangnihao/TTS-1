"""Tacotron Decoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom

from seq2seq.layers import PreNet


class DynamicConvolutionAttention(nn.Module):
    """Dynamic convolutional attention
    """
    def __init__(self, query_dim, attn_dim, static_channels,
                 static_kernel_size, dynamic_channels, dynamic_kernel_size,
                 prior_length, alpha, beta):
        """Instantiate the attention layer
        """
        super().__init__()

        self.query_dim = query_dim
        self.attn_dim = attn_dim
        self.static_channels = static_channels
        self.static_kernel_size = static_kernel_size
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_length = prior_length
        self.alpha = alpha
        self.beta = beta

        self.query_layer = nn.Linear(in_features=query_dim,
                                     out_features=attn_dim)

        self.key_layer = nn.Linear(attn_dim,
                                   dynamic_channels * dynamic_kernel_size,
                                   bias=False)

        self.static_conv_filters = nn.Conv1d(
            in_channels=1,
            out_channels=static_channels,
            kernel_size=static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False)

        self.static_filter_layer = nn.Linear(in_features=static_channels,
                                             out_features=attn_dim,
                                             bias=False)
        self.dynamic_filter_layer = nn.Linear(in_features=dynamic_channels,
                                              out_features=attn_dim)

        self.v = nn.Linear(in_features=attn_dim, out_features=1, bias=False)

        prior = betabinom.pmf(range(prior_length), prior_length - 1, alpha,
                              beta)
        self.register_buffer("prior", torch.FloatTensor(prior).flip(0))

    def forward(self, query, attention_weights):
        """Forward pass
        """
        prior_filter = F.conv1d(
            F.pad(attention_weights.unsqueeze(1), (self.prior_length - 1, 0)),
            self.prior.view(1, 1, -1))
        prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)

        G = self.key_layer(torch.tanh(self.query_layer(query)))

        # Compute dynamic filters
        dynamic_filter = F.conv1d(attention_weights.unsqueeze(0),
                                  G.view(-1, 1, self.dynamic_kernel_size),
                                  padding=(self.dynamic_kernel_size - 1) // 2,
                                  groups=query.size(0))
        dynamic_filter = dynamic_filter.view(query.size(0),
                                             self.dynamic_channels,
                                             -1).transpose(1, 2).contiguous()

        # Compute static filters
        static_filter = self.static_conv_filters(
            attention_weights.unsqueeze(1)).transpose(1, 2).contiguous()

        # Compute alignment
        alignment = self.v(
            torch.tanh(
                self.static_filter_layer(static_filter) +
                self.dynamic_filter_layer(dynamic_filter))).squeeze(
                    -1) + prior_filter

        # Compute attention weights
        attention_weights = F.softmax(alignment, dim=-1)

        return attention_weights


class Decoder(nn.Module):
    """Tacotron decoder
    """
    def __init__(self, n_mels, memory_dim, prenet_layer_sizes, dropout,
                 attn_dim, static_channels, static_kernel_size,
                 dynamic_channels, dynamic_kernel_size, prior_length, alpha,
                 beta, attn_rnn_size, decoder_rnn_size, r):
        """Instantiate the Decoder
        """
        super().__init__()

        self.n_mels = n_mels
        self.memory_dim = memory_dim
        self.prenet_layer_sizes = prenet_layer_sizes
        self.dropout = dropout
        self.attn_dim = attn_dim
        self.static_channels = static_channels
        self.static_kernel_size = static_kernel_size
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_length = prior_length
        self.alpha = alpha
        self.beta = beta
        self.decoder_rnn_size = decoder_rnn_size
        self.r = r

        # Prenet
        self.prenet = PreNet(in_dim=n_mels,
                             layer_sizes=prenet_layer_sizes,
                             dropout=dropout)

        # Dynamic Convolutional Attention
        self.attention = DynamicConvolutionAttention(
            query_dim=attn_rnn_size,
            attn_dim=attn_dim,
            static_channels=static_channels,
            static_kernel_size=static_kernel_size,
            dynamic_channels=dynamic_channels,
            dynamic_kernel_size=dynamic_kernel_size,
            prior_length=prior_length,
            alpha=alpha,
            beta=beta)

        # Attention RNN
        self.attention_rnn = nn.LSTMCell(input_size=memory_dim +
                                         prenet_layer_sizes[-1],
                                         hidden_size=attn_rnn_size)

        # Decoder projections
        self.decoder_projection = nn.Linear(in_features=memory_dim +
                                            decoder_rnn_size,
                                            out_features=decoder_rnn_size)

        # Decoder RNNs
        self.decoder_rnn1 = nn.LSTMCell(input_size=decoder_rnn_size,
                                        hidden_size=decoder_rnn_size)
        self.decoder_rnn2 = nn.LSTMCell(input_size=decoder_rnn_size,
                                        hidden_size=decoder_rnn_size)

        # Output layer
        self.output_layer = nn.Linear(in_features=decoder_rnn_size,
                                      out_features=n_mels * r,
                                      bias=False)

    def forward(self, y, memory, attention_weights, attention_context,
                attn_rnn_hx, decoder_rnn1_hx, decoder_rnn2_hx):
        """Forward pass
        """
        B, N = y.size()

        # Prenet
        y = self.prenet(y)

        # Attention RNN output using previous timestep context
        attn_rnn_h, attn_rnn_c = self.attention_rnn(
            torch.cat((y, attention_context), dim=-1), attn_rnn_hx)
        # Apply dropout
        attn_rnn_h = F.dropout(attn_rnn_h,
                               p=self.dropout,
                               training=self.training)

        # Compute current timestep attention weights and context
        attention_weights = self.attention(attn_rnn_h, attention_weights)
        attention_context = torch.matmul(attention_weights.unsqueeze(1),
                                         memory).squeeze(1)

        # Decoder RNNs
        x = self.linear(torch.cat((attention_context, attn_rnn_h), dim=-1))

        decoder_rnn1_h, decoder_rnn1_c = self.decoder_rnn1(x, decoder_rnn1_hx)
        decoder_rnn1_h = F.dropout(decoder_rnn1_h,
                                   p=self.dropout,
                                   training=self.training)
        x = x + decoder_rnn1_h

        decoder_rnn2_h, decoder_rnn2_c = self.decoder_rnn2(x, decoder_rnn2_hx)
        decoder_rnn2_h = F.dropout(decoder_rnn2_h,
                                   p=self.dropout,
                                   training=self.training)
        x = x + decoder_rnn2_h

        # Output layer
        y = self.output_layer(x).view(B, N, 2)

        return y, attention_weights, attention_context, (
            attn_rnn_h, attn_rnn_c), (decoder_rnn1_h,
                                      decoder_rnn1_c), (decoder_rnn2_h,
                                                        decoder_rnn2_c)
