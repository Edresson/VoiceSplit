# VoiceSplit

Final project for **SCC5830- Image Processing** @ ICMC/USP.

* 11572715 - Edresson Casanova 
* 8531702 - Pedro Regattieri Rocha 

## Abstract
This project aims to develop a system that, given an audio file, is able to separate overlapping voices based on the speaker's speech characteristics. For this we will use Honey Spectrograms. This system can be used in many applications, for example to improve the quality of automatic speech recognition in noisy environments. The final application that we think for this system is the separation of speakers for the generation of datasets. The SPIRA Project (
Early detection system for respiratory failure through audio analysis) that brings together researchers from USP and aims to build a method for screening people with COVID-19 based on audio and artificial intelligence, started collection in the ICUs. Obtaining data from patients infected with COVID-19, however, as most of these patients are elderly, many have difficulties reading the phrases proposed in the study, and the doctor who is carrying out the collection helps him. In this way, in most audio files there is the voice of the doctor who is doing the collection, his voice is not desired, and it can lead the learning models to overfit, so it is necessary to remove it.

## Development
The neural architecture will be based on VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking, proposed by Google researchers. However, we intend to explore the use of QRNNs (Quasi-Recurrent Neural Networks) instead of LSTM (Long Short Term Memory) and explore the use of the recent Mish activation function. QRNNs, are based on convolutional neural networks and have proven to be faster and to achieve results superior to LSTMs in various tasks of Natural Language Processing. On the other hand, the Mish activation function (https://arxiv.org/abs/1908.08681) has brought significant improvements to the ImageNet challenge and as far as we know its effectiveness has not yet been proven in speech data. Although spectrograms are images, the domain is different and some techniques do not have the same efficiency. To train the model we need a way to represent the speaker for the model. This is usually done by embeddings of the speakers extracted from a system of verification/identification of speakers. To do so, we initially intend to use the Speech2Phone (https://arxiv.org/abs/2002.11213), which is a multilingual and text-independent speaker identification system. The official implementation of Speech2Phone is open source and available at: https://github.com/Edresson/Speech2Phone

## Dataset
For the task we intend to use the LibreSpeech dataset initially. However, to use it in this task, we need to generate audios with overlappings voices.


## Current ToDos: 
*  code trainner and dataset load
*  code preprocessing in LibreSpeech following models/voicefilter/data-LibreSpeech/README.md and generic soluction for other datasets.

* import QRNNs from official repository: https://github.com/salesforce/pytorch-qrnn
    * see QRNN+DRELU activation:  https://arxiv.org/pdf/1707.08214v2.pdf we going implemente this??
    * Propouse QRNN+MISH ?? if yes we need compare in other task because  train in this task is very slow ... I think its very interesting because QRNN is fully convoluctional and MISH is very good for conv layers ... 

* implemente Mish activate function (its very easy)

* Implement Powerlaw compression loss following:https://github.com/stegben/voicefilter/tree/powerlaw-compression-loss

* try Speech2Phone if Speech2Phone dont Work well use a GE2E (the same speaker encoder used on Google's paper), this implementation is very good https://github.com/Edresson/GE2E-Speaker-Encoder and use CorentinJ pretrained model.
