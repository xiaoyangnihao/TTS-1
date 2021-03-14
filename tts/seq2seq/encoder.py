"""Tacotron Encoder"""

import torch
import torch.nn as nn
from tts.layers.common import BatchNormConv1D, Highway, PreNet


class CBHG(nn.Module):
    """Bank of 1-D Convolutions + Highway networks + residual connections + Bidirectional GRU
    """
    def __init__(self, in_dim, K, convbank_channels, projection_channels,
                 num_highway_layers, highway_layer_size, gru_size):
        """Instantiate the CBHG block
        """
        super().__init__()

        self.in_dim = in_dim
        self.K = K
        self.convbank_channels = convbank_channels
        self.projection_channels = projection_channels
        self.num_highway_layers = num_highway_layers
        self.highway_layer_size = highway_layer_size
        self.gru_size = gru_size

        # Bank of 1-D Convolutions
        self.conv_bank = nn.ModuleList([
            BatchNormConv1D(in_channels=in_dim,
                            out_channels=convbank_channels,
                            kernel_size=k,
                            stride=1,
                            padding=[(k - 1) // 2, k // 2],
                            activation=nn.ReLU()) for k in range(1, K + 1)
        ])

        # 1-D Convolutional projections
        sizes = [K * convbank_channels] + projection_channels
        activations = [nn.ReLU()] * (len(projection_channels) - 1) + [None]
        self.conv_projections = nn.ModuleList([
            BatchNormConv1D(in_size,
                            out_size,
                            kernel_size=3,
                            stride=1,
                            padding=[1, 1],
                            activation=act)
            for in_size, out_size, act in zip(sizes, sizes[1:], activations)
        ])

        # Highway layer
        self.pre_highway = nn.Linear(projection_channels[-1],
                                     highway_layer_size,
                                     bias=False)

        self.highways = nn.ModuleList(
            [Highway(highway_layer_size) for _ in range(num_highway_layers)])

        # Bidirectional GRU
        self.gru = nn.GRU(highway_layer_size,
                          gru_size,
                          1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, inputs):
        """Forward pass
        """
        x = inputs

        # 1-D Convolution bank
        outs = [conv(x) for conv in self.conv_bank]
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.convbank_channels * len(self.conv_bank)

        # 1-D Convolution projections
        for conv in self.conv_projections:
            x = conv(x)

        # Residual connections
        x += inputs

        x = x.transpose(1, 2).contiguous()

        # Highway layers
        if self.highway_layer_size != self.conv_projections[-1]:
            x = self.pre_highway(x)
        for highway in self.highways:
            x = highway(x)

        # Bidirectional GRU
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        return outputs


class Encoder(nn.Module):
    """Tacotron encoder
    """
    def __init__(self, char_embedding_dim, prenet_layer_sizes, dropout, K,
                 convbank_channels, projection_channels, num_highway_layers,
                 highway_layer_size, gru_size):
        """Instantiate the encoder
        """
        super().__init__()

        self.char_embedding_dim = char_embedding_dim
        self.prenet_layer_sizes = prenet_layer_sizes
        self.dropout = dropout
        self.K = K
        self.convbank_channels = convbank_channels
        self.projection_channels = projection_channels
        self.num_highway_layers = num_highway_layers
        self.highway_layer_size = highway_layer_size
        self.gru_size = gru_size

        self.prenet = PreNet(in_dim=char_embedding_dim,
                             layer_sizes=prenet_layer_sizes,
                             dropout=dropout)

        self.cbhg = CBHG(in_dim=prenet_layer_sizes[-1],
                         K=K,
                         convbank_channels=convbank_channels,
                         projection_channels=projection_channels,
                         num_highway_layers=num_highway_layers,
                         highway_layer_size=highway_layer_size,
                         gru_size=gru_size)

    def forward(self, inputs):
        """Forward pass
        """
        inputs = self.prenet(inputs)
        inputs = self.cbhg(inputs.transpose(1, 2).contiguous())

        return inputs
