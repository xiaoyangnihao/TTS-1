"""Tacotron layers"""

import torch.nn as nn
import torch.nn.functional as F


class PreNet(nn.Module):
    """PreNet module
    """
    def __init__(self, in_size, layer_sizes, dropout):
        """Instantiate the PreNet
        """
        super().__init__()

        self.in_size = in_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        sizes = [in_size] + layer_sizes
        self.prenet_layers = nn.ModuleList([
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
            for in_dim, out_dim in zip(sizes, sizes[1:])
        ])

    def forward(self, x):
        """Forward pass
        """
        for layer in self.prenet_layers:
            x = F.dropout(F.relu(layer(x)),
                          p=self.dropout,
                          training=self.training)

        return x


class BatchNormConv1D(nn.Module):
    """1-D Convolution with BatchNorm and Activation
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=True):
        """Instantiate the 1-D Convolution layer
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward pass
        """
        x = self.batchnorm(self.conv(x))

        x = nn.ReLU(x) if self.activation else x

        return x


class Highway(nn.Module):
    """Highway layer
    """
    def __init__(self, in_features, out_features):
        """Instantiate the Highway layer
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.H = nn.Linear(in_features=in_features, out_features=out_features)
        self.H.bias.data.zero_()

        self.T = nn.Linear(in_features=in_features, out_features=out_features)
        self.T.bias.data.fill_(-1)

    def forward(self, x):
        """Forward pass
        """
        H = nn.ReLU(self.H(x))
        T = nn.Sigmoid(self.T(x))

        return H * T + x * (1.0 - T)
