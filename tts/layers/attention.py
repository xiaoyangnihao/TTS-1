"""Attention layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import betabinom


class DynamicConvolutionAttention(nn.Module):
    """Dynamic Convolutional Attention
    """
    def __init__(self,
                 query_dim,
                 memory_dim,
                 attn_dim,
                 static_channnels,
                 static_kernel_size,
                 dynamic_channels,
                 dynamic_kernel_size,
                 prior_len=11,
                 alpha=0.1,
                 beta=0.9):
        """Instantiate the Attention layer
        """
        super().__init__()

        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.attn_dim = attn_dim
        self.static_channels = static_channnels
        self.static_kernel_size = static_kernel_size
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_len = prior_len
        self.alpha = alpha
        self.beta = beta

        self.attention_weights = None
        self._mask_value = 1e-8

        self.query_layer = nn.Linear(query_dim, attn_dim)
        self.key_layer = nn.Linear(attn_dim,
                                   dynamic_channels * dynamic_kernel_size,
                                   bias=False)

        self.static_filter = nn.Conv1d(1,
                                       static_channnels,
                                       static_kernel_size,
                                       padding=(static_kernel_size - 1) // 2,
                                       bias=False)

        self.static_filter_layer = nn.Linear(static_channnels,
                                             attn_dim,
                                             bias=False)
        self.dynamic_filter_layer = nn.Linear(dynamic_channels, attn_dim)

        self.v = nn.Linear(attn_dim, 1, bias=False)

        prior = betabinom.pmf(range(prior_len), prior_len - 1, alpha, beta)
        self.register_buffer("prior", torch.FloatTensor(prior).flip(0))

    def _init_states(self, memory):
        """Initialize the attention states
        """
        B, T = memory.size(0), memory.size(1)

        self.attention_weights = torch.zeros([B, T], device=memory.device)
        self.attention_weights[:, 0] = 1

    def forward(self, query, memory, mask):
        """Forward pass
        """
        # Compute prior filter
        prior_filter = F.conv1d(
            F.pad(self.attention_weights.unsqueeze(1),
                  (self.prior_len - 1, 0)), self.prior.view(1, 1, -1))
        prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)

        # Compute dynamic filter
        G = self.key_layer(torch.tanh(self.query_layer(query)))
        dynamic_filter = F.conv1d(self.attention_weights.unsqueeze(0),
                                  G.view(-1, 1, self.dynamic_kernel_size),
                                  padding=(self.dynamic_kernel_size - 1) // 2,
                                  groups=query.size(0))
        dynamic_filter = dynamic_filter.view(query.size(0),
                                             self.dynamic_channels,
                                             -1).transpose(1, 2).contiguous()

        # Compute static filter
        static_filter = self.static_filter(
            self.attention_weights.unsqueeze(1)).transpose(1, 2).contiguous()

        # Compute alignment and attention weights
        alignment = self.v(
            torch.tanh(
                self.static_filter_layer(static_filter) +
                self.dynamic_filter_layer(dynamic_filter))).squeeze(
                    -1) + prior_filter
        attention_weights = F.softmax(alignment, dim=-1)

        # Apply masking
        if mask is not None:
            attention_weights.data.masked_fill_(~mask, self._mask_value)
        self.attention_weights = attention_weights

        # Compute attention context
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

        return context
