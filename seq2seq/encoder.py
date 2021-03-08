"""Tacotron Encoder"""

import torch
import torch.nn as nn

from seq2seq.layers import BatchNormConv1D, Highway, PreNet


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
                            padding=k // 2,
                            activation=nn.ReLU()) for k in range(1, K + 1)
        ])

        # Max pooling
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # 1-D Convolutional projections
        sizes = [K * convbank_channels] + projection_channels
        activations = [nn.ReLU()] * (len(projection_channels) - 1) + [None]
        self.conv_projections = nn.ModuleList([
            BatchNormConv1D(in_size,
                            out_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
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

    def forward(self, x):
        """Forward pass
        """
        residual = x

        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2).contiguous()

        T = x.size(-1)

        # Bank of 1-D Convolutions
        x = torch.cat(
            [conv_layer(x)[:, :, :T] for conv_layer in self.conv_bank], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv_bank)

        # Max pooling
        x = self.max_pool(x)[:, :, :T]

        # 1-D Convolution projections
        for conv_layer in self.conv_projections:
            x = conv_layer(x)

        x = x.transpose(1, 2).contiguous()

        if x.size(-1) != self.highway_layer_size:
            x = self.pre_highway(x)

        # Residual connection + Highway layers
        x += residual
        for highway_layer in self.highways:
            x = highway_layer(x)

        # Bidirectional GRU
        outputs, _ = self.gru(x)

        return outputs


class Encoder(nn.Module):
    """Tacotron encoder
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

        self.embedding_layer = nn.Embedding(num_embeddings=num_chars,
                                            embedding_dim=char_embedding_dim)

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

    def forward(self, x):
        """Forward pass
        """
        x = self.embedding_layer(x)
        x = self.prenet(x)
        x = self.cbhg(x)

        return x
