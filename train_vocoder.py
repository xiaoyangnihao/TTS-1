"""Vocoder training"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import config.config as cfg
from vocoder.dataset import VocoderDataset
from vocoder.model import WaveRNN


def save_checkpoint(checkpoint_dir, model, optimizer, scheduler, step):
    """Write checkpoint to disk
    """
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_step{step:09d}.pth")

    torch.save(checkpoint_state, checkpoint_path)
    print(f"Written checkpoint: {checkpoint_path} to disk")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load the checkpoint from the disk
    """
    print(f"Loading checkpoint: {checkpoint_path} from disk")

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["step"]


def train_model(train_data_dir, checkpoint_dir, resume_checkpoint_path=None):
    """Train the model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Specify the device to train on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model
    model = WaveRNN(
        n_mels=cfg.audio["n_mels"],
        hop_length=cfg.audio["hop_length"],
        num_bits=cfg.audio["n_bits"],
        audio_embedding_dim=cfg.vocoder_model["audio_embedding_dim"],
        conditioning_rnn_size=cfg.vocoder_model["conditioning_rnn_size"],
        rnn_size=cfg.vocoder_model["rnn_size"],
        fc_size=cfg.vocoder_model["fc_size"])
    model = model.to(device)
    model.train()

    # Instantiate the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.vocoder_training["learning_rate"])

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, cfg.vocoder_training["lr_scheduler_step_size"],
        cfg.vocoder_training["lr_scheduler_gamma"])

    if resume_checkpoint_path is not None:
        global_step = load_checkpoint(resume_checkpoint_path, model, optimizer,
                                      scheduler)
    else:
        global_step = 0

    # Instantiate the dataloader
    dataset = VocoderDataset(train_data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.vocoder_training["batch_size"],
                            shuffle=True,
                            num_workers=cfg.vocoder_training["num_workers"],
                            pin_memory=True,
                            drop_last=True)

    num_epochs = cfg.vocoder_training["num_steps"] // len(dataloader) + 1
    start_epoch = global_step // len(dataloader) + 1

    for epoch in range(start_epoch, num_epochs + 1):
        avg_loss = 0

        for idx, (mels, qwavs) in enumerate(dataloader, 1):
            # Place tensors on the appropriate device
            mels, qwavs = mels.to(device), qwavs.to(device)

            optimizer.zero_grad()

            wav_hat = model(qwavs[:, :-1], mels)
            loss = F.cross_entropy(wav_hat.transpose(1, 2), qwavs[:, 1:])

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            avg_loss += (loss.item() - avg_loss) / idx

            if global_step % cfg.checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir, model, optimizer, scheduler,
                                global_step)

        print(
            f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Current lr: {scheduler.get_last_lr()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the vocoder model")

    parser.add_argument(
        "--data_dir",
        help="Path to processed dataset to be used to train the model",
        required=True)

    parser.add_argument(
        "--checkpoint_dir",
        help="Path to location where training checkpoints will be saved",
        required=True)

    parser.add_argument(
        "--resume_checkpoint_path",
        help="If specified load checkpoint and resume training from that point"
    )

    args = parser.parse_args()

    train_data_dir = args.train_data_dir
    checkpoint_dir = args.checkpoint_dir
    resume_checkpoint_path = args.resume_checkpoint_path

    train_model(train_data_dir, checkpoint_dir, resume_checkpoint_path)
