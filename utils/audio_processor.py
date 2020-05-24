import torch
import torch.utils.data
from utils.audio import WaveGlowSTFT as STFT
from utils.generic_utils import load_wav_to_torch
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0

class AudioProcessor(object):
    from utils.audio import WaveGlowSTFT as STFT
    def __init__(self, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, n_mel_channels=80,num_freq=None):
        self.stft = STFT(filter_length=filter_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    sampling_rate=sampling_rate,
                                    mel_fmin=mel_fmin, 
                                    mel_fmax=mel_fmax,
                                    n_mel_channels=n_mel_channels)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_mel_from_audio_path(self, audio_path):
        audio, sampling_rate = load_wav_to_torch(audio_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        melspectrogram = self.get_mel(audio)
        return melspectrogram