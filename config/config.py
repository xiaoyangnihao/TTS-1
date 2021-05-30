"""Configuration Parameters"""

dataset = "LJSpeech"  # The name of the dataset

text_processor = "en"  # Currently supported -> english text: "en" and indic language text: "indic"

# Audio processing configuration
audio = {
    "sampling_rate": 22050,  # Sampling rate of the wav files in the dataset
    "max_db": 100,
    "ref_db": 20,
    "n_fft": 2048,
    "win_length": 1100,  # 50 ms window length: sampling_rate * 50 / 1000
    "hop_length": 275,  # 12.5 ms frame shift: sampling_rate * 12.5 / 1000
    "n_mels": 80,
    "fmin": 50,
    "n_bits": 10,  # The bit depth of the signal (used in the vocoder)
}

# TTS configuration
tts_model = {
    "char_embedding_dim": 256,
    "prenet_layer_sizes": [256, 128],
    "dropout": 0.5,
    "zoneout": 0.1,

    # CBHG Encoder
    "encoder": {
        "K": 16,
        "convbank_channels": 128,
        "projection_channels": [128, 128],
        "num_highway_layers": 4,
        "highway_layer_size": 128,
        "gru_size": 128
    },

    # Dynamic convolutional attention
    "attention": {
        "attn_dim": 128,
        "static_channels": 8,
        "static_kernel_size": 21,
        "dynamic_channels": 8,
        "dynamic_kernel_size": 21,
        "prior_length": 11,
        "alpha": 0.1,
        "beta": 0.9,
    },

    # Autoregressive decoder
    "decoder": {
        "attn_rnn_size": 256,
        "decoder_rnn_size": 256,
        "reduction_factor": 2,
    }
}

tts_training = {
    "batch_size": 32,
    "bucket_size_multiplier": 5,
    "num_steps": 250000,
    "checkpoint_interval": 10000,
    "num_workers": 8,
    "clip_grad_norm": 0.05,
    "learning_rate": 1e-3,
    "lr_scheduler_milestones": [20000, 40000, 100000, 150000, 200000],
    "lr_scheduler_gamma": 0.5,
}

# Vocoder configuration
vocoder_model = {
    "audio_embedding_dim": 256,
    "conditioning_rnn_size": 128,
    "rnn_size": 896,
    "fc_size": 1024,
}

vocoder_training = {
    "batch_size": 32,
    "num_steps": 250000,
    "sample_frames": 24,
    "learning_rate": 4e-4,
    "lr_scheduler_step_size": 25000,
    "lr_scheduler_gamma": 0.5,
    "checkpoint_interval": 10000,
    "num_workers": 8,
}
