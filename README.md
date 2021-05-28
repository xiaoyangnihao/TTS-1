# Recurrent Neural Network based Text-to-Speech systems

This repository contains code to train a End-to-End Speech Synthesis system. Both single speaker and multi-speaker models are supported.

Currrently, the text frontend supports English, as well as the following Indic languages: Assamese, Bengali, Gujarati, Hindi, Marathi, Rajasthani, Tamil, Telugu. In the case of Indic languages, the text is first transformed from unicode to [Common Label Set](https://www.iitm.ac.in/donlab/tts/downloads/cls/cls_v2.1.6.pdf), which provides a common representation for similar sounds across all Indic languages, before being used in the remainder of the voice building process. A parser to convert Indic text from unicode to common label set has been provided as part of the repository.
 
The system consists of two parts:

1. A Tacotron model with Dynamic Convolutional Attention which modifies the hybrid location sensitive attention mechanism to be purely location based as described in [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf), resulting in better generalization on long utterances. This model takes text (in the form of character sequence) as input and predicts a sequence of mel-spectrogram frames as output (the seq2seq model).

2. A WaveRNN based vocoder; which takes the mel-spectrogram predicted in the previous step as input and generates a waveform as output (the vocoder model).

All audio processing parameters, model hyperparameters, training configuration etc are specified in the `config/config.py` folder. 

Both the seq2seq model and the vocoder model need to be trained seperately. Training using automatic mixed precision is supported.

# Quick start
## Train TTS from scratch

1. Download and extract dataset 
    
    1. English single speaker dataset [LJSpeech](https://keithito.com/LJ-Speech-Dataset/):

        ```bash
        wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
        tar -xvjf LJSpeech-1.1.tar.bz2
        ```
    
    2. Hindi single speaker dataset [IIITH-CVIT-IndicSpeech](http://cvit.iiit.ac.in/research/projects/cvit-projects/text-to-speech-dataset-for-indian-languages):
       
       Request access to the dataset by filling in the google form available at the webpage. Once access has been granted, download and extract the dataset
       
       Once the dataset has been downloaded and extracted, convert the text prompts from unicode (Hindi) to ITRANS

        ```python
        python unicode_to_cls_converter.py \
            --unicode_prompts_file <Path to the file containing unicode text prompts> \
            --itrans_prompts_file <Path to the output file> \
            --lang_code <Code representing the Indic language ("hi" in this case)>
        ```

2. Edit the configuration parameters in `config/config.py` appropriate for the dataset to be used for training

3. Process the downloaded dataset, and split into train and eval splits 

    ```python
    python preprocess.py \
            --dataset_dir <Path to the root of the downloaded dataset> \
            --out_dir <Output path to write the processed dataset>
    ```

4. Train the Tacotron (seq2seq) model

    ```python
    python train_tts.py \
            --train_data_dir <Path to the processed train split> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --alignments_dir <Path to the location where training alignments will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

5. Train the vocoder model

    ```python
    python train_vocoder.py \
            --train_data_dir <Path to the processed train split> \
            --checkpoint_dir <Path to location where training checkpoints will be saved> \
            --resume_checkpoint_path <If specified load checkpoint and resume training>
    ```

## Synthesize using a trained TTS model
1. Prepare the text to be synthesized
    
    The text to be synthesized should be placed in the `synthesis.csv` file in the following format

    ```
    ID_1|TEXT_1
    ID_2|TEXT_2
    .
    .
    .
    ```

2. Text to speech synthesis
    
    ```python
    python tts_synthesis.py \
            --synthesis_file <Path to the synthesis.csv file (created in Step 1)> \
            --seq2seq_checkpoint <Path to the trained seq2seq model to use for synthesis> \
            --vocoder_checkpoint <Path to the trained vocoder model to use for synthesis> \
            --out_dir <Path to where the synthesized waveforms will be written to disk>
    ```
## Acknowledgements

This code is based on the code in the following repositories
1. [bshall/Tacotron](https://github.com/bshall/Tacotron)
2. [mozilla/TTS](https://github.com/mozilla/TTS)

## References

1. [Location Relative Attention Mechanisms for Robust Long-Form Speech Synthesis](https://arxiv.org/pdf/1910.10288.pdf)
2. [Tacotron: Towards End-To-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf)
3. [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf)
