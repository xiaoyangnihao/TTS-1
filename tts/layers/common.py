"""Common layers"""

import torch.nn as nn


class PreNet(nn.Module):
    """Prenet module
    """
    def __init__(self, in_dim, layer_sizes, dropout):
        """Instantiate the PreNet
        """
        super().__init__()

        self.in_dim = in_dim
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        sizes = [in_dim] + layer_sizes
        self.prenet_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(sizes, sizes[1:])
        ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass
        """
        for layer in self.prenet_layers:
            x = self.dropout(self.activation(layer(x)))

        return x


class BatchNormConv1D(nn.Module):
    """1-D Convolution + Activation + Batch Normalization
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation=None):
        """Instantiate the layer
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=kernel_size // 2,
                              bias=False)

        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """Forward pass
        """
        x = self.conv(x)

        if self.activation is not None:
            x = self.activation(x)

        x = self.batchnorm(x)

        return x


class Highway(nn.Module):
    """Highway layer
    """
    def __init__(self, layer_size):
        """Instantiate the layer
        """
        super().__init__()

        self.H = nn.Linear(layer_size, layer_size)
        self.H.bias.data.zero_()

        self.T = nn.Linear(layer_size, layer_size)
        self.T.bias.data.fill_(-1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass
        """
        H = self.relu(self.H(x))
        T = self.sigmoid(self.T(x))

        return H * T + x * (1.0 - T)
