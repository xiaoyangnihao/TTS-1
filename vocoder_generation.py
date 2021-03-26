"""Vocoder copy-synthesis generation (Generate a waveform given a mel-spectrogram) on held-out eval set"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch

import config.config as cfg
from vocoder.model import WaveRNN


def _load_filenames(filename):
    """Load list of filenames in held-out eval set
    """
    with open(filename, "r") as file_reader:
        data = file_reader.readlines()

    data = [instance.strip("\n") for instance in data]

    data = [instance.split("|") for instance in data]

    filenames = [instance[0] for instance in data]

    return filenames


def generate(checkpoint_path, held_out_filenames, held_out_mel_dir, out_dir):
    """Generate waveforms from mel-spectrogram using vocoder model
    """
    os.makedirs(out_dir, exist_ok=True)

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the vocoder model
    model = WaveRNN(
        n_mels=cfg.audio["n_mels"],
        hop_length=cfg.audio["hop_length"],
        num_bits=cfg.audio["n_bits"],
        audio_embedding_dim=cfg.vocoder_model["audio_embedding_dim"],
        conditioning_rnn_size=cfg.vocoder_model["conditioning_rnn_size"],
        rnn_size=cfg.vocoder_model["rnn_size"],
        fc_size=cfg.vocoder_model["fc_size"])
    model = model.to(device)
    model.eval()

    # Restore the vocoder model to specified training checkpoint
    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model_step = checkpoint["step"]

    # Generate waveforms for all files in held-out eval set
    for filename in held_out_filenames:
        print(f"Generating waveform from mel-spectrogram for: {filename}")

        mel = np.load(os.path.join(held_out_mel_dir, filename + ".npy"))
        mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)

        # Generate waveform from mel-spectrogram
        with torch.no_grad():
            wav_hat = model.generate(mel)

        # Write the generated wavform to disk
        out_path = os.path.join(out_dir,
                                f"model_step{model_step:09d}_{filename}.wav")
        sf.write(out_path, wav_hat, cfg.audio["sampling_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vocoder generation (waveform generation) on held-out set")

    parser.add_argument(
        "--checkpoint_path",
        help="Path to the checkpoint to use to instantiate the model",
        required=True)

    parser.add_argument(
        "--eval_data_dir",
        help="Path to the dir containing the held-out eval data",
        required=True)

    parser.add_argument(
        "--out_dir",
        help="Path to dir where generated waveforms will be written to disk",
        required=True)

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    eval_data_dir = args.eval_data_dir
    out_dir = args.out_dir

    held_out_filenames = _load_filenames(
        os.path.join(eval_data_dir, "metadata_eval.txt"))
    held_out_mel_dir = os.path.join(eval_data_dir, "mel")

    generate(checkpoint_path, held_out_filenames, held_out_mel_dir, out_dir)
