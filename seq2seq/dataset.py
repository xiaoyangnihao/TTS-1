"""Tacotron dataset"""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as samplers
from numpy.lib.arraypad import pad
from text.english import load_cmudict, symbol_to_id, text_to_id
from torch.nn.utils.rnn import pad_sequence
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


class SortedSampler(samplers.Sampler):
    """Sorted sampler
    """
    def __init__(self, data, sort_key):
        """Instantiate the sampler
        """
        super().__init__(data)

        self.data = data
        self.sort_key = sort_key

        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1], reverse=True)
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(samplers.BatchSampler):
    """Bucket batch sampler
    """
    def __init__(self, sampler, sort_key, batch_size, bucket_size_multiplier,
                 drop_last):
        """Instantiate the Sampler
        """
        super().__init__(sampler, batch_size, drop_last)

        self.sort_key = sort_key
        self.bucket_sampler = samplers.BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)),
            False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in samplers.SubsetRandomSampler(
                    list(
                        samplers.BatchSampler(sorted_sampler, self.batch_size,
                                              self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


class TTSDataset(Dataset):
    """TTS dataset
    """
    def __init__(self, train_data_dir):
        """Instantiate the dataset
        """
        super().__init__()

        self.train_data_dir = train_data_dir

        self.training_instances = _load_training_instances(
            os.path.join(train_data_dir, "metadata_train.txt"))
        self.lengths = [instance[2] for instance in self.training_instances]

        self.max_length_index = np.argmax(self.lengths)

        self.cmudict = load_cmudict()

    def sort_key(self, index):
        return self.lengths[index]

    def __len__(self):
        return len(self.training_instances)

    def __getitem__(self, index):
        filename, text = self.training_instances[index][
            0], self.training_instances[index][1]

        mel_path = os.path.join(self.train_data_dir, "mel", filename + ".npy")
        mel = np.load(mel_path)

        text = text_to_id(text, self.cmudict)

        return (torch.FloatTensor(mel).transpose_(0, 1),
                torch.LongTensor(text), index == self.max_length_index)


def collate(batch, reduction_factor=2):
    """Collate and create padded batches
    """
    mels, texts, attn_flags = zip(*batch)

    mels, texts = list(mels), list(texts)

    # Pad the batch to a muliple of reduction factor
    if len(mels[0]) % reduction_factor != 0:
        mels[0] = F.pad(mels[0], (0, 0, 0, reduction_factor - 1))

    mel_lengths = [len(mel) for mel in mels]
    text_lengths = [len(text) for text in texts]

    mels = pad_sequence(mels, batch_first=True)
    texts = pad_sequence(texts,
                         batch_first=True,
                         padding_value=symbol_to_id["_PAD_"])
    attn_flags = [idx for idx, flag in enumerate(attn_flags) if flag]

    return texts, text_lengths, mels.transpose_(1, 2), mel_lengths, attn_flags
