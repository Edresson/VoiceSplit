# VoiceSplit

Final project for **SCC5830- Image Processing** @ ICMC/USP.

* 11572715 - Edresson Casanova 
* 8531702 - Pedro Regattieri Rocha 

## Abstract
This project’s goal is the development of a system that, given an audio input, is able to, through the use of Mel Spectrograms, filter overlapping voices based on each speaker’s speech patterns. 
This system may then be used to, for example, improve the quality of automatic speech recognition systems used in noisy environments. 
Our main objective for this system is the separation of people having a conversation into different entries to help generate data sets. 
The SPIRA Project (An early detection system that checks for respiratory failure through audio analysis and the use of artificial intelligence) has brought together USP researchers with the purpose of creating a method to screen suspected cases of COVID-19.
Researchers from SPIRA have already started collecting voice samples from COVID-19 patients in Intensive Care Units. This process can be challenging as many of these patients are elderly or may otherwise have some difficulty repeating the requested phrases, requiring the help of the doctor collecting the audio sample, meaning some files will also contain the voice of the doctor doing the sampling, which may in turn cause the learning models to overfit. As such it becomes necessary to separate the doctor’s voice from the patient’s sample to preserve the integrity of the data.

## Development
The neural architecture will be based on VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking, proposed by Google researchers. However, we intend to explore the use of QRNNs (Quasi-Recurrent Neural Networks) instead of LSTM (Long Short Term Memory) and explore the use of the recent Mish activation function. QRNNs, are based on convolutional neural networks and have proven to be faster and to achieve results superior to LSTMs in various tasks of Natural Language Processing. On the other hand, the Mish activation function (https://arxiv.org/abs/1908.08681) has brought significant improvements to the ImageNet challenge and as far as we know its effectiveness has not yet been proven in speech data. Although spectrograms are images, the domain is different and some techniques do not have the same efficiency. To train the model we need a way to represent the speaker for the model. This is usually done by embeddings of the speakers extracted from a system of verification/identification of speakers. To do so, we initially intend to use the Speech2Phone (https://arxiv.org/abs/2002.11213), which is a multilingual and text-independent speaker identification system. The official implementation of Speech2Phone is open source and available at: https://github.com/Edresson/Speech2Phone

## Dataset
For the task we intend to use the LibreSpeech dataset initially. However, to use it in this task, we need to generate audios with overlappings voices.


## Current ToDos:

* ** WaveGlow works in 22khz :( and for comparison it VoiceFilter we need train in LibriSpeech and LibreSpeech is 16khz**:
    * For fix this i need support a WaveRNN universal (https://github.com/mozilla/TTS/issues/221).
    * Look soluction for support two backends WaveRNN and WaveGlow (implemented)

* code dataloader (partial fixed, because LibriSpeech have 16kHz sample rate and universal WaveGlow is treined in 22khz) 
* code tests from dataloader (v)
* code trainner (v)
* code preprocessing (generate data) in LibreSpeech following models/voicefilter/data-LibreSpeech/README.md and generic soluction for other datasets. For details check the voicefilter paper and https://google.github.io/speaker-id/publications/VoiceFilter/

* see this paper https://link.springer.com/chapter/10.1007/978-3-030-37731-1_3 for ideas, its new state of the art.



* import QRNNs from official repository: https://github.com/salesforce/pytorch-qrnn
    * see QRNN+DRELU activation:  https://arxiv.org/pdf/1707.08214v2.pdf we going implemente this??
    * Propouse QRNN+MISH ?? if yes we need compare in other task because  train in this task is very slow ... I think its very interesting because QRNN is fully convoluctional and MISH is very good for conv layers ... 

* implemente Mish activate function (its very easy)

* Use NovoGrad optimizer https://arxiv.org/abs/1905.11286 ?? They is better than adam, is used in Jasper the state of art in Automatic Speech Recognition 

* Implement Powerlaw compression loss following:https://github.com/stegben/voicefilter/tree/powerlaw-compression-loss (v)

* Use loss function in two phases, get mixed spec and subtract the output spectrogram, in theory the new spectrogram have a others sounds without voice of target spectogram. Calculate loss in this spectogram and sum with real loss. In final train we get a two spectrograms with good quality. 

* use a multi loss ?? mybe Powerlaw compression loss with   Si-SNR (pit_criterion.py)  with L1 loss ??  L1 loss is used in text2Speech, and in text2peech the speech is very clean.  L1 loss with Powerlaw compression loss ??? L1 loss is better than MSE in Text2Speech.


* try Speech2Phone if Speech2Phone dont Work well use a GE2E (the same speaker encoder used on Google's paper), this implementation is very good https://github.com/Edresson/GE2E-Speaker-Encoder and use CorentinJ pretrained model.




## Future Features

* Two Universal Neural Vocoders is supported (WaveGlow and WaveRNN)
* Two Neural Speakers encoders is supported (GE2E and Speech2Phone)