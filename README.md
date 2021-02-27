# Recurrent Neural Network based Text-to-Speech systems

This repository contains code to train a End-to-End Speech Synthesis system, based on the Tactoron2 model with modifications as described in [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf).

The system consists of two parts:  
	1. A Tacotron2 model with Dynamic Convolutional Attention which modifies the hybrid location sensitive attention mechanism to be purely location based, resulting in better generalization on long utterances. This model takes text (in the form of character sequence) as input and predicts a sequence of mel-spectrogram frames as output.  
	2. A WaveRNN based vocoder; which takes the mel-spectrogram predicted in the previous step as input and generates a waveform as output.  

To train the model we will be using the LJSpeech dataset, which is a single speaker English dataset consisting of ~24 hrs of speech and corresponding text transcripts. All model details, training configuration etc are specified in `config.yaml`

## Getting started
### 0. Download dataset and make train/eval split

	1. Download and extract the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/):
		```
		wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
		tar -xvjf LJSpeech-1.1.tar.bz2

		```  
	2. Split the metadata.csv into train and eval subsets respectively metadata\_train.csv and metadata\_eval.csv.
		```
		shuf metadata.csv > metadata_shuf.csv
		head -n 12000 metadata_shuf.csv > metadata_train.csv
		tail -n 1100 metadata_shuf.csv > metadata_val.csv

		```
### 1. Preprocessing

COMING SOON

### 2. Training

COMING SOON

### 3. Generation

COMING SOON

## Acknowledgements

The code in this repository is based on the code in the following repositories
1. [bshall/Tacotron](https://github.com/bshall/Tacotron)

## References

1. [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf)
2. [Tacotron: Towards End-To-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf)
3. [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)
