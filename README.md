# VoiceSplit

Pytorch unofficial implementation of [VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking](https://arxiv.org/abs/1810.04826)


Final project for **SCC5830- Image Processing** @ ICMC/USP.

## Dataset
For the task we intend to use the LibriSpeech dataset initially. However, to use it in this task, we need to generate audios with overlappings voices.

## Improvements
    We use Si-SNR with PIT instead of Power Law compressed loss, because it allows us to achieve a better result ( comparison available in: https://github.com/Edresson/VoiceSplit).
    We used the MISH activation function instead of ReLU and this has improved the result

# Report
You can see a report of what was done in this repository [here](https://github.com/Edresson/VoiceSplit/blob/master/Final%20Report.pdf)

## Demos
Colab notebooks Demos:

Exp 1: [link](https://colab.research.google.com/drive/1GljrJOo_uMRfSUDIIht1Eo1HqaAkhY36?usp=sharing)

Exp 2: [link](https://colab.research.google.com/drive/19Rh4YaZtcI2gSAvI9q40YUXubxOXLkUG?usp=sharing)

Exp 3: [link](https://drive.google.com/file/d/1b0gCNazK6exulR765PAI3xkMMXzBR-Fp/view?usp=sharing)

Exp 4: [link](https://colab.research.google.com/drive/1AJ0VS8_Vv3Ayph20WFyi6iWASEJpQE3N?usp=sharing)

Exp 5 (best): [link](https://colab.research.google.com/drive/1FYlaQX4XzN4W_6JB4z9R5SFrBF8a9RqC?usp=sharing)

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
