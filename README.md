# VoiceSplit

Pytorch unofficial implementation of [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)


Final project for **SCC5830- Image Processing** @ ICMC/USP.

## Dataset
For the task we intend to use the LibreSpeech dataset initially. However, to use it in this task, we need to generate audios with overlappings voices.

## Improvements
    We use Si-SNR with PIT instead of Power Law compressed loss, because it allows us to achieve a better result ( comparison available in: https://github.com/Edresson/VoiceSplit).
    We used the MISH activation function instead of ReLU and this has improved the result

# Report
You can see a report of what was done in this repository [here](https://github.com/Edresson/VoiceSplit/blob/master/Final%20Report.pdf)

## Demos
Colab notebooks Demos:

Exp 1: https://shorturl.at/eBX18

Exp 2: https://shorturl.at/oyEJN

Exp 3: https://shorturl.at/blnEW

Exp 4: https://shorturl.at/qFJN8

Exp 5 (best): https://shorturl.at/kvAQ8

Site demo for the experiment with best results (Exp 5): https://edresson.github.io/VoiceSplit/
## ToDos:
Create documentation for the repository and
remove unused code

## Future Works

* Train VoiceSplit model with GE2E3k and Mean Squared Error loss function

## Acknowledgment:
In this repository it contains codes of other collaborators, the due credits were given in the used functions:

Preprocessing: Eren Gölge @erogol

VoiceFilter Model: Seungwon Park @seungwonpark
