"""Tacotron Encoder"""

import torch
import torch.nn as nn

from seq2seq.layers import BatchNormConv1D, Highway, PreNet


class CBHG(nn.Module):
    """1-D Convolution Bank + Highway Network + Residual Connections + Bidirectional GRU
    """
    def __init__(self, input_channels, K, convbank_channels, conv_projections,
                 num_highway_layers, highway_layer_size, gru_size):
        """Instantiate the CBHG layer
        """
        super().__init__()

        self.input_channels = input_channels
        self.K = K
        self.convbank_channels = convbank_channels
        self.conv_projections = conv_projections
        self.num_highway_layers = num_highway_layers
        self.highway_layer_size = highway_layer_size
        self.gru_size = gru_size

        # 1-D Convolution Bank
        self.convbank = nn.ModuleList([
            BatchNormConv1D(in_channels=input_channels,
                            out_channels=convbank_channels,
                            kernel_size=k) for k in range(1, K + 1)
        ])

        # Max pooling
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        # 1-D Convolutional Projections
        channels = [K * convbank_channels] + conv_projections
        activations = [True] * (len(conv_projections) - 1) + [False]
        self.conv_projections = nn.ModuleList([
            BatchNormConv1D(in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=3,
                            activation=activation) for in_size, out_size,
            activation in zip(channels, channels[1:], activations)
        ])

        # Linear projection to highway
        self.prehighway_projection = nn.Linear(
            in_features=conv_projections[-1],
            out_features=highway_layer_size,
            bias=False) if conv_projections[-1] != highway_layer_size else None

        # Highway layer
        self.highway = nn.ModuleList([
            Highway(in_features=highway_layer_size,
                    out_features=highway_layer_size)
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
        x = x.transpose(1, 2).contiguous()

        T = x.size(-1)
        residual = x

        # 1-D Convolution bank
        x = [conv(x)[:, :, :T] for conv in self.convbank]
        x = torch.cat(x, dim=1)

        # Maxpool
        x = self.maxpool(x)

        # 1-D Convolution projections
        x = self.conv_projections(x[:, :, :T])

        # Residual connections
        x = x + residual
        x = x.transpose(1, 2).contiguous()

        # Highway layers
        if self.prehighway_projection is not None:
            x = self.prehighway_projection(x)
        x = self.highway(x)

        # Bidirectional GRU
        x, _ = self.gru(x)

        return x


class TacotronEncoder(nn.Module):
    """Tacotron seq2seq encoder
    """
    def __init__(self, num_chars, char_embedding_dim, prenet_layer_sizes,
                 dropout, cbhg_K, cbhg_convbank_channels,
                 cbhg_conv_projections, cbhg_num_highway_layers,
                 cbhg_highway_layer_size, cbhg_gru_size):
        """Instantiate the encoder
        """
        super().__init__()

        self.num_chars = num_chars
        self.char_embedding_dim = char_embedding_dim
        self.prenet_layer_sizes = prenet_layer_sizes
        self.dropout = dropout
        self.cbhg_K = cbhg_K
        self.cbhg_convbank_channels = cbhg_convbank_channels
        self.cbhg_conv_projections = cbhg_conv_projections
        self.cbhg_num_highway_layers = cbhg_num_highway_layers
        self.cbhg_highway_layer_size = cbhg_highway_layer_size
        self.cbhg_gru_size = cbhg_gru_size

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_chars,
                                            embedding_dim=char_embedding_dim)

        # Prenet
        self.prenet = PreNet(in_size=char_embedding_dim,
                             layer_sizes=prenet_layer_sizes,
                             dropout=dropout)

        # CBHG
        self.cbhg = CBHG(input_channels=prenet_layer_sizes[-1],
                         K=cbhg_K,
                         convbank_channels=cbhg_convbank_channels,
                         conv_projections=cbhg_conv_projections,
                         num_highway_layers=cbhg_num_highway_layers,
                         highway_layer_size=cbhg_highway_layer_size,
                         gru_size=cbhg_gru_size)

    def forward(self, x):
        """Forward pass
        """
        x = self.embedding_layer(x)
        x = self.prenet(x)
        x = self.cbhg(x)

        return x
