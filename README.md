# Recurrent Neural Network based Text-to-Speech systems

This repository contains code to train a End-to-End Speech Synthesis system, based on the Tactoron2 model with modifications as described in [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf).
 
The system consists of two parts:

1. A Tacotron model with Dynamic Convolutional Attention which modifies the hybrid location sensitive attention mechanism to be purely location based, resulting in better generalization on long utterances. This model takes text (in the form of character sequence) as input and predicts a sequence of mel-spectrogram frames as output (the seq2seq model).

2. A WaveRNN based vocoder; which takes the mel-spectrogram predicted in the previous step as input and generates a waveform as output (the vocoder model).

All audio processing parameters, model hyperparameters, training configuration etc are specified in `config/config.py`. Both the Tacotron seq2seq model and the WaveRNN based vocoder model are trained on a single GPU, using automatic mixed precision.
# Quick start

## Train TTS from scratch
1. Download dataset

    Download and extract the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset:
    
    ```bash
    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -xvjf LJSpeech-1.1.tar.bz2
    ```  

2. Preprocess the downloaded dataset and perform feature extraction on the wav files

    ```python
    python preprocess.py \
            --dataset_dir <Path to the root of the downloaded dataset> \
            --out_dir <Output path to write the processed dataset>
    ```

3. Split the processed dataset into train and eval subsets (split metadata.txt into metadata\_train.txt and metadata\_eval.txt respectively).
		
    ```bash
    shuf metadata.txt > metadata_shuf.txt
    head -n 12000 metadata_shuf.txt > metadata_train.txt
    tail -n 1100 metadata_shuf.txt > metadata_eval.txt
    ```

4. Train the seq2seq model

    ```python
    python train_tts.py \
            --data_dir <Path to the processed dataset to be used to train the model> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --alignments_dir <Path to the location where training alignments will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

5. Train the vocoder model

    ```python
    python train_vocoder \
            --data_dir <Path to the processed dataset to be used to train the model> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

## Acknowledgements

This code is based on the code in the following repositories
1. [bshall/Tacotron](https://github.com/bshall/Tacotron)
2. [mozilla/TTS](https://github.com/mozilla/TTS)

## References

1. [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf)
2. [Tacotron: Towards End-To-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf)
3. [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)
