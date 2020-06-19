import os
import torch
import torch.utils.data
import scipy
import librosa
import pickle
import copy
import numpy as np
import soundfile as sf
from utils.audio import WaveGlowSTFT
from utils.generic_utils import load_wav_to_torch
from scipy.io.wavfile import read
from pprint import pprint
from scipy import signal, io
import torchaudio

MAX_WAV_VALUE = 32768.0

class WrapperAudioProcessor(object):
    def __init__(self, config_audio):
        # get backend
        backend = config_audio['backend']
        mel_spec = config_audio["mel_spec"]
        if backend == 'waveglow':
            self.ap = WaveGlowAudioProcessor(**config_audio[backend], mel_spec=mel_spec)
        elif backend == 'wavernn':
            self.ap = WaveRNNAudioProcessor(**config_audio[backend], mel_spec=mel_spec)
        elif backend == 'voicefilter':
            self.ap = openVoiceFilterAudioProcessor(**config_audio[backend])
        else:
            raise ValueError("Invalid AudioProcessor Backend: ")

    def inv_spectrogram(self, spectrogram, phase=None):
        return self.ap.inv_spectrogram(spectrogram, phase)

    def torch_inv_spectrogram(self, spectrogram, phase=None):
        return self.ap.torch_inv_spectrogram(spectrogram, phase)
    def get_spec_from_audio_path(self, audio_path):
        return self.ap.get_spec_from_audio_path(audio_path)

    def get_spec_from_audio(self, audio, return_phase=False):
        spec = self.ap.get_spec_from_audio(audio)
        if isinstance(spec, tuple) and return_phase:
            return self.ap.get_spec_from_audio(audio)
        elif isinstance(spec, tuple) and  not return_phase:
            return self.ap.get_spec_from_audio(audio)[0]
        elif not isinstance(spec, tuple) and return_phase:
            return self.ap.get_spec_from_audio(audio), None

        return spec

    def save_wav(self, wav, path):
        return self.ap.save_wav(wav, path)

    def load_wav(self, path):
        return self.ap.load_wav(path)

    def get_mel(self, wav):
        return self.ap.get_mel(wav)

class WaveRNNAudioProcessor(object):
    """
    This class was taken from Eren Gölge implementation of WaveRNN (https://github.com/erogol/WaveRNN). 
    So the credits are from @erogol
    """
    def __init__(self,
                 bits=None,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=None,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 mel_spec=True,
                 force_convert_SR=False,
                 **kwargs):

        print(" > Setting up Audio Processor...")

        self.bits = bits
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = 0 if mel_fmin is None else mel_fmin
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        self.mel_spec = mel_spec
        self.force_convert_SR = force_convert_SR
        print(" | > Audio Processor attributes.")
        members = vars(self)
        for key, value in members.items():
            print("   | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (MAX_WAV_VALUE / max(0.01, np.max(np.abs(wav))))
        io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _build_mel_basis(self, ):
        n_fft = (self.num_freq - 1) * 2
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        if self.signal_norm:
            S_norm = ((S - self.min_level_db) / - self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm :
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm) 
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                    self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        print(" | > fft size: {}, hop length: {}, win length: {}".format(
            n_fft, hop_length, win_length))
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S).T

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S).T

    def inv_spectrogram(self, spectrogram, phase=None):
        if self.mel_spec:
            inv = self.inv_mel_spectrogram(spectrogram)
        else:
            inv = self.inv_linear_spectrogram(spectrogram)
        return inv
    def get_spec_from_audio_path(self, path):
        wav = self.load_wav(path)
        if self.mel_spec:
            spec = self.melspectrogram(wav)
        else:
            spec = self.spectrogram(wav)
        return spec.astype(np.float32)

    def get_spec_from_audio(self, wav):
        if self.mel_spec:
            spec = self.melspectrogram(wav)
        else:
            spec = self.spectrogram(wav)
        return spec.astype(np.float32)


    def inv_linear_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        spectrogram = spectrogram.T
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        mel_spectrogram = mel_spectrogram.T
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        else:
            return self._griffin_lim(S**self.power)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.1 sec margin """
        margin = int(self.sample_rate * 0.1)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=40, frame_length=1024, hop_length=256)[0]

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2 ** qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal)

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2 ** qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x

    def load_wav(self, filename, sr=None):
        if self.force_convert_SR:
            sr = self.sample_rate
        if sr is None:
            x, sr = sf.read(filename)
        else:
            x, sr = librosa.load(filename, sr=sr)
        

        if self.do_trim_silence:
            x = self.trim_silence(x)
        assert self.sample_rate == sr, "%s vs %s"%(self.sample_rate, sr)
        return x

    def encode_16bits(self, x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    def quantize(self, x):
        return (x + 1.) * (2**self.bits - 1) / 2

    def dequantize(self, x):
        return 2 * x / (2**self.bits - 1) - 1


class WaveGlowAudioProcessor(object):
    """
    This class was removed from the official WaveGlow repository (https://github.com/NVIDIA/waveglow)
    so this part of the code is under the license imposed on the repository. And the credits go to your developers. 
    """
    def __init__(self, segment_length, filter_length,
                 hop_length, win_length, sample_rate, mel_fmin, mel_fmax, n_mel_channels=80,num_freq=None, power=None, griffin_lim_iters=None, mel_spec=None):
        self.stft = WaveGlowSTFT(filter_length=filter_length,
                                    hop_length=hop_length,
                                    win_length=win_length,
                                    sampling_rate=sample_rate,
                                    mel_fmin=mel_fmin, 
                                    mel_fmax=mel_fmax,
                                    n_mel_channels=n_mel_channels)
        self.segment_length = segment_length
        self.sampling_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self.mel_spec = mel_spec
        self.n_fft = (num_freq - 1) * 2

    def get_mel(self, audio_norm):
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        return melspec.T

    def get_mag(self, audio_norm):
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        magspec = self.stft.mag_spectrogram(audio_norm)
        magspec = torch.squeeze(magspec, 0)
        return magspec.T

    def get_spec_from_audio(self, audio):
        if self.mel_spec:
            spectrogram = self.get_mel(audio)
        else:
            spectrogram = self.get_mag(audio)
        return spectrogram

    def get_spec_from_audio_path(self, audio_path):
        audio = self.load_wav(audio_path)
        if self.mel_spec:
            spectrogram = self.get_mel(audio)
        else:
            spectrogram = self.get_mag(audio)
        return spectrogram

    def mag_to_mel(self, y):
        return self.stft.mag_to_mel_spectrogram(y.T).T

    # Griffin-Lim
    def inv_spectrogram(self, spectrogram, phase=None):
        """Converts spectrogram to waveform using librosa"""
        spectrogram = spectrogram.T
        S = self.stft.spectral_de_normalize(spectrogram).cpu().detach().numpy()
        if self.mel_spec:
            mel_basis = self.stft.mel_basis.cpu().detach().numpy()
            S = self._mel_to_linear(S, mel_basis) 
        # Reconstruct phase
        return self._griffin_lim(S**self.power)

    def _mel_to_linear(self, mel_spec, mel_basis):
        return np.maximum(1e-10, np.dot(np.linalg.pinv(mel_basis), mel_spec))

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

    def load_wav(self,audio_path):
        audio, sampling_rate = load_wav_to_torch(audio_path)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        return audio / MAX_WAV_VALUE

class openVoiceFilterAudioProcessor():
    #  This class is adapted from VoiceFilter open source implementation (https://github.com/mindslab-ai/voicefilter/blob/master/utils/audio.py)
    # So Credits for: @seungwonpark
    def __init__(self, sample_rate, n_fft, num_freq, hop_length, win_length, preemphasis, power, min_level_db, ref_level_db, num_mels, griffin_lim_iters):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_freq = num_freq
        self.hop_length = hop_length
        self.win_length = win_length
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.num_mels = num_mels
        self.preemphasis = preemphasis
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters

        self.mel_basis = librosa.filters.mel(sr=self.sample_rate,
                                             n_fft=self.n_fft,
                                             n_mels=self.num_mels)

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=self.n_fft,
                              hop_length=self.hop_length,
                              win_length=self.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2spec(self, y):
        audio_class = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)#, window_fn=torch.hamming_window(self.n_fft,periodic=False, alpha=0.5, beta=0.5))
        self.torch_spec = audio_class(torch.from_numpy(y))
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T # to make [time, freq]
        return S, D

    def istft_phase(self, mag, phase):
        stft_matrix = mag * np.exp(1j*phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hop_length,
                             win_length=self.win_length)
    def spec2wav(self, spectrogram, phase=None):
        if phase is not None:
            #print("Using Mixed Phase for spec2wav")
            spectrogram, phase = spectrogram.T, phase.T
            # used during inference only
            # spectrogram: enhanced output
            # phase: use noisy input's phase, so no GLA is required
            S = self.db_to_amp(self.denormalize(spectrogram) + self.ref_level_db)
            return self.istft_phase(S, phase)
        else:
            #print("Using GL for spec2wav")
            spectrogram = spectrogram.T
            S = self.db_to_amp(self.denormalize(spectrogram) + self.ref_level_db)
            return self._griffin_lim(S**self.power)

    def torch_spec2wav(self, spectrogram, phase=None):
        spectrogram = spectrogram.transpose(2,1)
        phase = phase.transpose(2,1)
        # denormalise spectrogram
        S =  (torch.clamp(spectrogram, 0.0, 1.0) - 1.0) * -self.min_level_db
        S = S + self.ref_level_db
        # db_to_amp
        stft_matrix = torch.pow(10.0, S * 0.05)
        # invert phase
        phase = torch.stack([phase.cos(), phase.sin()], dim=-1).to(dtype=stft_matrix.dtype, device=stft_matrix.device)
        stft_matrix = stft_matrix.unsqueeze(-1).expand_as(phase)
        return torchaudio.functional.istft(stft_matrix * torch.exp(phase), self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=torch.hamming_window(self.win_length, periodic=False, alpha=0.5, beta=0.5).to(device=stft_matrix.device), center=True, normalized=False, onesided=True, length=None)

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.min_level_db
    
    def inv_spectrogram(self, spectrogram, phase=None):
        return self.spec2wav(spectrogram, phase)

    def torch_inv_spectrogram(self, spectrogram, phase=None):
        return self.torch_spec2wav(spectrogram, phase)

    def get_spec_from_audio_path(self, audio_path):
        return self.wav2spec(self.load_wav(audio_path))

    def get_spec_from_audio(self, audio):
        return self.wav2spec(audio)

    def save_wav(self, wav, path):
        wav_norm = wav * (MAX_WAV_VALUE / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def load_wav(self, path):
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav