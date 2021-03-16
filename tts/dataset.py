"""Tacotron dataset"""

import os

import numpy as np
import torch
from text.english import load_cmudict, symbol_to_id, text_to_id
from torch.utils.data import Dataset


def _load_training_instances(filename):
    """Load the training instances from disk
    """
    with open(filename, "r") as file_reader:
        training_instances = file_reader.readlines()

    training_instances = [
        instance.strip("\n") for instance in training_instances
    ]

    training_instances = [
        instance.split("|") for instance in training_instances
    ]

    return training_instances


def _pad_text(text_seq, max_len):
    """Pad a 1-D sequence
    """
    return np.pad(text_seq, (0, max_len - len(text_seq)),
                  mode="constant",
                  constant_values=symbol_to_id["_PAD_"])


def _pad_mel(mel, max_len):
    """Pad a 2-D sequence
    """
    x = np.pad(mel, [(0, max_len - len(mel)), (0, 0)],
               mode="constant",
               constant_values=0)

    return x


class TTSDataset(Dataset):
    """TTS dataset
    """
    def __init__(self, train_data_dir, reduction_factor):
        """Instantiate the dataset
        """
        super().__init__()

        self.train_data_dir = train_data_dir
        self.reduction_factor = reduction_factor

        self.training_instances = _load_training_instances(
            os.path.join(train_data_dir, "metadata_train.txt"))
        self.lengths = [instance[2] for instance in self.training_instances]

        self.max_length_index = np.argmax(self.lengths)

        self.cmudict = load_cmudict()

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, index):
        filename, text = self.training_instances[index][
            0], self.training_instances[index][1]

        mel_path = os.path.join(self.train_data_dir, "mel", filename + ".npy")
        mel = np.load(mel_path)

        text = text_to_id(text, self.cmudict)

        return text, mel.T

    def collate(self, batch):
        """Collate and create padded batches
        """
        text_lengths = [len(x[0]) for x in batch]
        mel_lengths = [len(x[1]) for x in batch]

        max_text_len = np.max(text_lengths)
        max_mel_len = np.max(mel_lengths) + 1

        if max_mel_len % self.reduction_factor != 0:
            max_mel_len += self.reduction_factor - max_mel_len % self.reduction_factor
            assert max_mel_len % self.reduction_factor == 0

        texts = np.array([_pad_text(x[0], max_text_len) for x in batch],
                         dtype=np.int)
        texts = torch.LongTensor(texts)
        text_lengths = torch.LongTensor(text_lengths)

        mels = np.array([_pad_mel(x[1], max_mel_len) for x in batch],
                        dtype=np.float32)
        mels = torch.FloatTensor(mels)
        mel_lengths = torch.LongTensor(mel_lengths)

        return texts, text_lengths, mels, mel_lengths
