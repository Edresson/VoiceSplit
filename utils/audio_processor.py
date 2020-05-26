import torch
import torch.utils.data
import numpy as np
from utils.audio import WaveGlowSTFT
from utils.generic_utils import load_wav_to_torch
from scipy.io.wavfile import read
import scipy
import librosa
MAX_WAV_VALUE = 32767.0

class AudioProcessor(object):
    def __init__(self, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax, n_mel_channels=80,num_freq=None, power=None, griffin_lim_iters=None, mel_spec=None):
        self.stft = WaveGlowSTFT(filter_length=filter_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    sampling_rate=sampling_rate,
                                    mel_fmin=mel_fmin, 
                                    mel_fmax=mel_fmax,
                                    n_mel_channels=n_mel_channels)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self.mel_spec = mel_spec
        self.n_fft = (num_freq - 1) * 2

    def get_mel(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def get_mag(self, audio):
        audio_norm = audio / MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        magspec = self.stft.mag_spectrogram(audio_norm)
        magspec = torch.squeeze(magspec, 0)
        return magspec

    def get_spec_from_audio_path(self, audio_path):
        audio, sampling_rate = load_wav_to_torch(audio_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        if self.mel_spec:
            spectrogram = self.get_mel(audio)
        else:
            spectrogram = self.get_mag(audio)
        return spectrogram

    def mag_to_mel(self, y):
        return self.stft.mag_to_mel_spectrogram(y)

    # Griffin-Lim
    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        # torch.from_numpy()
        S = self.stft.spectral_de_normalize(spectrogram).cpu().detach().numpy()
        if self.mel_spec:
            mel_basis = self.stft.mel_basis.cpu().detach().numpy()
            S = self._mel_to_linear(S, mel_basis) 
        # Reconstruct phase
        return self._griffin_lim(S**self.power)

    def _mel_to_linear(self, mel_spec, mel_basis):
        return np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec))

    """ def inv_melspectrogram(self, mel_spectrogram):
        '''Converts melspectrogram to waveform using librosa'''
        # torch.from_numpy()
        S = self.stft.spectral_de_normalize(mel_spectrogram).cpu().detach().numpy()
        mel_basis = self.stft.mel_basis.cpu().detach().numpy()
        S = self._mel_to_linear(S, mel_basis)  # Convert back to linear
        return self._griffin_lim(S**self.power)"""

    def _librosa_istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def _librosa_stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode='constant'
        )

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._librosa_istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._librosa_stft(y)))
            y = self._librosa_istft(S_complex * angles)
        return y

    def save_wav(self, wav, path):
        wav_norm = wav * (MAX_WAV_VALUE / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sampling_rate, wav_norm.astype(np.int16))