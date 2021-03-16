"""Tacotron model"""

import config.config as cfg
import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.seq2seq.decoder import Decoder
from tts.seq2seq.encoder import Encoder


class Tacotron(nn.Module):
    """Tacotron model
    """
    def __init__(self, num_chars):
        """Instantiate the Tacotron model
        """
        super().__init__()

        self.num_chars = num_chars
        self.cfg = cfg
        self.n_mels = cfg.audio["n_mels"]
        self.attn_rnn_size = cfg.tts_model["decoder"]["attn_rnn_size"]
        self.decoder_rnn_size = cfg.tts_model["decoder"]["decoder_rnn_size"]
        self.memory_dim = 2 * cfg.tts_model["encoder"]["gru_size"]
        self.reduction_factor = cfg.tts_model["decoder"]["reduction_factor"]

        # Tacotron seq2seq Encoder
        self.encoder = Encoder(
            num_chars=num_chars,
            char_embedding_dim=cfg.tts_model["char_embedding_dim"],
            prenet_layer_sizes=cfg.tts_model["prenet_layer_sizes"],
            dropout=cfg.tts_model["dropout"],
            K=cfg.tts_model["encoder"]["K"],
            convbank_channels=cfg.tts_model["encoder"]["convbank_channels"],
            projection_channels=cfg.tts_model["encoder"]
            ["projection_channels"],
            num_highway_layers=cfg.tts_model["encoder"]["num_highway_layers"],
            highway_layer_size=cfg.tts_model["encoder"]["highway_layer_size"],
            gru_size=cfg.tts_model["encoder"]["gru_size"])

        # Decoder
        self.decoder = Decoder(
            n_mels=cfg.audio["n_mels"],
            memory_dim=2 * cfg.tts_model["encoder"]["gru_size"],
            prenet_layer_sizes=cfg.tts_model["prenet_layer_sizes"],
            dropout=cfg.tts_model["dropout"],
            attn_rnn_size=cfg.tts_model["decoder"]["attn_rnn_size"],
            attn_dim=cfg.tts_model["attention"]["attn_dim"],
            static_channels=cfg.tts_model["attention"]["static_channels"],
            static_kernel_size=cfg.tts_model["attention"]
            ["static_kernel_size"],
            dynamic_channels=cfg.tts_model["attention"]["dynamic_channels"],
            dynamic_kernel_size=cfg.tts_model["attention"]
            ["dynamic_kernel_size"],
            prior_len=cfg.tts_model["attention"]["prior_length"],
            alpha=cfg.tts_model["attention"]["alpha"],
            beta=cfg.tts_model["attention"]["beta"],
            decoder_rnn_size=cfg.tts_model["decoder"]["decoder_rnn_size"],
            reduction_factor=cfg.tts_model["decoder"]["reduction_factor"])

    def forward(self, texts, mels):
        """Forward pass
        """
        # Group multiple frames in mels as per reduction factor
        mels = mels.view(mels.size(0), -1,
                         mels.size(-1) // self.reduction_factor)

        assert mels.size(1) == self.n_mels * self.reduction_factor

        B, N, T = mels.size()
        mels = mels.unbind(-1)

        # Encoder output
        memory = self.encoder(texts)

        # Initialize attention states
        attention_weights = F.one_hot(
            torch.zeros(B, dtype=torch.long, device=texts.device),
            memory.size(1)).float()

        attention_context = torch.zeros(B,
                                        self.memory_dim,
                                        device=texts.device)

        # Initialize attention RNN states
        attn_rnn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=texts.device),
            torch.zeros(B, self.attn_rnn_size, device=texts.device),
        )

        # Initialize decoder RNN stats
        decoder_rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=texts.device),
            torch.zeros(B, self.decoder_rnn_size, device=texts.device),
        )
        decoder_rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=texts.device),
            torch.zeros(B, self.decoder_rnn_size, device=texts.device),
        )

        # Initialize all zeros go frame to use as decoder input for first timestep
        go_frame = torch.zeros(B, N, device=texts.device)

        ys, attention = [], []
        for t in range(0, T):
            # Teacher forcing (use the previous timestep ground truth as input for the current timestep)
            y = mels[t - 1] if t > 0 else go_frame

            y, attention_weights, attention_context, attn_rnn_hx, decoder_rnn1_hx, decoder_rnn2_hx = self.decoder(
                y, memory, attention_weights, attention_context, attn_rnn_hx,
                decoder_rnn1_hx, decoder_rnn2_hx)

            ys.append(y)
            attention.append(attention_weights)

        ys = torch.cat(ys, dim=-1)
        ys = ys.view(B, self.n_mels, -1)

        attention = torch.stack(attention, dim=2)

        return ys, attention

    def generate(self, text, max_length=10000, stop_threshold=-0.2):
        """Generate a mel spectrogram from text
        """
        memory = self.encoder(text)
        B, T, _ = memory.size()

        attention_weights = F.one_hot(
            torch.zeros(B, dtype=torch.long, device=text.device), T).float()
        attention_context = torch.zeros(B, self.memory_dim, device=text.device)

        # Initialize attention RNN states
        attn_rnn_hx = (
            torch.zeros(B, self.attn_rnn_size, device=text.device),
            torch.zeros(B, self.attn_rnn_size, device=text.device),
        )

        # Initialize decoder RNN stats
        decoder_rnn1_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=text.device),
            torch.zeros(B, self.decoder_rnn_size, device=text.device),
        )
        decoder_rnn2_hx = (
            torch.zeros(B, self.decoder_rnn_size, device=text.device),
            torch.zeros(B, self.decoder_rnn_size, device=text.device),
        )

        # Initialize all zeros go frame to use as decoder input for first timestep
        go_frame = torch.zeros(B,
                               self.n_mels * self.reduction_factor,
                               device=text.device)

        ys, attention = [], []
        for t in range(0, max_length):
            # Use previous timestep prediction as input for current timestep
            y = ys[-1] if t > 0 else go_frame

            y, attention_weights, attention_context, attn_rnn_hx, decoder_rnn1_hx, decoder_rnn2_hx = self.decoder(
                y, memory, attention_weights, attention_context, attn_rnn_hx,
                decoder_rnn1_hx, decoder_rnn2_hx)

            # Stopping criterion
            if torch.all(y[:, -1] > stop_threshold):
                break

            ys.append(y)
            attention.append(attention_weights)

        ys = torch.cat(ys, dim=-1)
        ys = ys.view(B, self.n_mels, -1)

        attention = torch.stack(attention, dim=2)

        return ys, attention
