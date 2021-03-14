"""Tacotron Encoder"""

import torch
import torch.nn as nn
from tts.layers.common import BatchNormConv1D, Highway, PreNet


class CBHG(nn.Module):
    """CBHG module
    """
    def __init__(self, in_channels, K, convbank_channels, projection_channels,
                 num_highway_layers, highway_layer_size, gru_size):
        """Instantiate  the CBHG module
        """
        super().__init__()

        self.in_channels = in_channels
        self.K = K
        self.convbank_channels = convbank_channels
        self.projection_channels = projection_channels
        self.num_highway_layers = num_highway_layers
        self.highway_layer_size = highway_layer_size
        self.gru_size = gru_size

        # Bank of 1-D Convolutions
        self.conv_bank = nn.ModuleList([
            BatchNormConv1D(in_channels=in_channels,
                            out_channels=convbank_channels,
                            kernel_size=k) for k in range(1, K + 1)
        ])

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # 1-D Convolutional projections
        channels = [K * convbank_channels] + projection_channels
        activations = [nn.ReLU()] * (len(projection_channels) - 1) + [None]
        self.conv_projections = nn.ModuleList([
            BatchNormConv1D(in_channels=in_dim,
                            out_channels=out_dim,
                            kernel_size=3,
                            activation=act) for in_dim, out_dim, act in zip(
                                channels, channels[1:], activations)
        ])

        #  Highway layers
        self.pre_highway = nn.Linear(
            projection_channels[-1], highway_layer_size, bias=False
        ) if projection_channels[-1] != highway_layer_size else None

        self.highways = nn.ModuleList([
            Highway(layer_size=highway_layer_size)
            for _ in range(num_highway_layers)
        ])

        # Bidirectional GRU
        self.gru = nn.GRU(highway_layer_size,
                          gru_size,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, x):
        """Forward pass
        """
        T = x.size(-1)

        residual = x

        # Bank of 1-D Convolutions
        x = torch.cat([conv(x)[:, :, :T] for conv in self.conv_bank], dim=1)
        assert x.size(1) == self.in_channels * len(self.conv_bank)

        # Max pooling
        x = self.max_pool(x)[:, :, :T]

        # 1-D Convolutional Projections
        for conv in self.conv_projections:
            x = conv(x)

        # Residual Connections
        x = x + residual
        x = x.transpose(1, 2).contiguous()

        # Higway networks
        if self.pre_highway is not None:
            x = self.pre_highway(x)
        for highway in self.highways:
            x = highway(x)

        # Bidirectional GRU
        x, _ = self.gru(x)

        return x


class Encoder(nn.Module):
    """Tacotron seq2seq encoder
    """
    def __init__(self, num_chars, char_embedding_dim, prenet_layer_sizes,
                 dropout, K, convbank_channels, projection_channels,
                 num_highway_layers, highway_layer_size, gru_size):
        """Instantiate the encoder
        """
        super().__init__()

        self.num_chars = num_chars
        self.char_embedding_dim = char_embedding_dim
        self.prenet_layer_sizes = prenet_layer_sizes
        self.dropout = dropout
        self.K = K
        self.convbank_channels = convbank_channels
        self.projection_channels = projection_channels
        self.num_highway_layers = num_highway_layers
        self.highway_layer_size = highway_layer_size
        self.gru_size = gru_size

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_chars,
                                            embedding_dim=char_embedding_dim)

        # Prenet
        self.prenet = PreNet(in_dim=char_embedding_dim,
                             layer_sizes=prenet_layer_sizes,
                             dropout=dropout)

        # CBHG
        self.cbhg = CBHG(in_channels=prenet_layer_sizes[-1],
                         K=K,
                         convbank_channels=convbank_channels,
                         projection_channels=projection_channels,
                         num_highway_layers=num_highway_layers,
                         highway_layer_size=highway_layer_size,
                         gru_size=gru_size)

    def forward(self, x):
        """Forward pass
        """
        x = self.embedding_layer(x)
        x = self.prenet(x)
        x = self.cbhg(x.transpose(1, 2).contiguous())

        return x
