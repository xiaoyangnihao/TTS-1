---
layout: page
title: Recurrent Neural Network based End-to-End Text-to-Speech Systems
tagline:
description: nil
---
End-to-End Speech Synthesis system, based on the Tactoron2 model with modifications as described in [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf)

The system consists of two parts:

1. A Tacotron model with Dynamic Convolutional Attention which modifies the hybrid location sensitive attention mechanism to be purely location based, resulting in better generalization on long utterances. This model takes text as input and predicts a sequence of mel-spectrogram frames as output (the seq2seq model).

2. A WaveRNN based vocoder; which takes the mel-spectrogram predicted in the previous step as input and generates a waveform as output (the vocoder model). 
The code to train this system can be found at [anandaswarup/TTS](https://github.com/anandaswarup/TTS)