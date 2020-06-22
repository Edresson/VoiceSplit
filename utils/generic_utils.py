import os
import torch
import json
import re
import datetime
import librosa

import torch.nn as nn
import numpy as np
import librosa.util as librosa_util

from scipy.signal import get_window
from scipy.io.wavfile import read
from mir_eval.separation import bss_eval_sources


from itertools import permutations
import torch.nn.functional as F

def mix_wavfiles(output_dir, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path):
    data_out_dir = output_dir
    emb_audio, _ = librosa.load(embedding_utterance_path, sr=sample_rate)
    clean_audio, _ = librosa.load(clean_utterance_path, sr=sample_rate)
    interference, _ = librosa.load(interference_utterance_path, sr=sample_rate)
    assert len(emb_audio.shape) == len(clean_audio.shape) == len(interference.shape) == 1, \
        'wav files must be mono, not stereo'

    # trim initial and end  wave file silence using librosa
    emb_audio, _ = librosa.effects.trim(emb_audio, top_db=20)
    clean_audio, _ = librosa.effects.trim(clean_audio, top_db=20)
    interference, _ = librosa.effects.trim(interference, top_db=20)

    # calculate frames using audio necessary for config.audio['audio_len'] seconds
    audio_len_seconds = int(sample_rate * audio_len)

    # if merged audio is shorter than audio_len_seconds, discard it
    if clean_audio.shape[0] < audio_len_seconds or interference.shape[0] < audio_len_seconds:
        return

    clean_audio = clean_audio[:audio_len_seconds]
    interference = interference[:audio_len_seconds]
    # merge audio
    mixed_audio = clean_audio + interference

    # normlise audio
    norm_factor = np.max(np.abs(mixed_audio)) * 1.1
    clean_audio = clean_audio/norm_factor
    interference = interference/norm_factor
    mixed_audio = mixed_audio/norm_factor

    # save normalized wave files and wav emb ref
    target_wav_path = glob_re_to_filename(data_out_dir, form['target_wav'], num)
    mixed_wav_path = glob_re_to_filename(data_out_dir, form['mixed_wav'], num)
    emb_wav_path = glob_re_to_filename(data_out_dir, form['emb_wav'], num)
    librosa.output.write_wav(emb_wav_path, emb_audio, sample_rate)
    librosa.output.write_wav(target_wav_path, clean_audio, sample_rate)
    librosa.output.write_wav(mixed_wav_path, mixed_audio, sample_rate)

    # extract and save spectrograms
    clean_spec, _ = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec, _ = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)

def glob_re_to_filename(dire, glob, num):
    return os.path.join(dire, glob.replace('*', '%06d' % num))

# losses 
class PowerLaw_Compressed_Loss(nn.Module):
    def __init__(self, power=0.3, complex_loss_ratio=0.113):
        super(PowerLaw_Compressed_Loss, self).__init__()
        self.power = power
        self.complex_loss_ratio = complex_loss_ratio
        self.criterion = nn.MSELoss()
        self.epsilon = 1e-16 # use epsilon for prevent  gradient explosion

    def forward(self, prediction, target, seq_len=None, spec_phase=None):
        # prevent NAN loss
        prediction = prediction + self.epsilon
        target = target + self.epsilon

        prediction = torch.pow(prediction, self.power)
        target = torch.pow(target, self.power)

        spec_loss = self.criterion(torch.abs(target), torch.abs(prediction))
        complex_loss = self.criterion(target, prediction)

        loss = spec_loss + (complex_loss * self.complex_loss_ratio)
        return loss

# adapted from https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, inp):
        '''
        Forward pass of the function.
        '''
        return inp * torch.tanh(F.softplus(inp))


# adpted from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/pit_criterion.py
def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask

class SiSNR_With_Pit(nn.Module):
    def __init__(self):
        super(SiSNR_With_Pit, self).__init__()
        self.epsilon = 1e-16 # use epsilon for prevent  gradient explosion
    def forward(self, source, estimate_source, source_lengths):
        """
        Calculate SI-SNR with PIT training.
        Args:
            source: [B, C, T], B is batch size
            estimate_source: [B, C, T]
            source_lengths: [B], each item is between [0, T]
        """
        # get wav from spec phase
        assert source.size() == estimate_source.size()
        
        B, C, T = source.size()
        # mask padding position along T
        mask = get_mask(source, source_lengths)
        estimate_source *= mask

        # Step 1. Zero-mean norm
        num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
        mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
        mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
        zero_mean_target = source - mean_target
        zero_mean_estimate = estimate_source - mean_estimate
        # mask padding position along T
        zero_mean_target *= mask
        zero_mean_estimate *= mask

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + self.epsilon  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + self.epsilon)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + self.epsilon)  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
        # one-hot, [C!, C, C]
        index = torch.unsqueeze(perms, 2)
        perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
        # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
        snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
        max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
        # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
        max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
        max_snr /= C
        loss = 20 - torch.mean(max_snr) # i use 20 because 20 is very high value
        return loss

def validation(criterion, ap, model, testloader, tensorboard, step, cuda=True, loss_name='si_snr', test=False):
    sdrs = []
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            emb, clean_spec, mixed_spec, clean_wav, mixed_wav, mixed_phase, seq_len = batch[0]

            emb = emb.unsqueeze(0)
            clean_spec = clean_spec.unsqueeze(0)
            mixed_spec = mixed_spec.unsqueeze(0)

            if cuda:
                emb = emb.cuda()
                clean_spec = clean_spec.cuda()
                mixed_spec = mixed_spec.cuda()

            est_mask = model(mixed_spec, emb)
            est_mag = est_mask * mixed_spec
            if loss_name == 'power_law_compression':
                test_loss = criterion(clean_spec, est_mag, seq_len).item()
            mixed_spec = mixed_spec[0].cpu().detach().numpy()
            clean_spec = clean_spec[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()

            est_wav = ap.inv_spectrogram(est_mag, phase=mixed_phase)
            est_mask = est_mask[0].cpu().detach().numpy()
            if loss_name == 'si_snr':
                test_loss = criterion(torch.from_numpy(np.array([[clean_wav]])), torch.from_numpy(np.array([[est_wav]])), seq_len).item()
            sdr = bss_eval_sources(clean_wav, est_wav, False)[0][0]
            if not test:
                tensorboard.log_evaluation(test_loss, sdr,
                                    mixed_wav, clean_wav, est_wav,
                                    mixed_spec.T, clean_spec.T, est_mag.T, est_mask.T,
                                    step)
                    print("Validation Loss:", test_loss)
                    print("Validation SDR:", sdr)
                break
            sdrs.append(sdr)
            losses.append(test_loss)
        if test:
            mean_test_loss = np.array(losses).mean()
            mean_sdr = np.array(sdrs).mean()
            print("Mean Test Loss:", mean_test_loss)
            print("Mean Test SDR:", mean_sdr)
            return mean_test_loss, mean_sdr
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def load_config_from_str(input_str):
    config = AttrDict()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config

def copy_config_file(config_file, out_path, new_fields):
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if isinstance(value, str):
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(out_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def set_init_dict(model_dict, checkpoint, c):
    """
    This Function is adpted from: https://github.com/mozilla/TTS
    Credits: @erogol
    """
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint['model'].items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in checkpoint['model'].items() if k in model_dict
    }
    # 2. filter out different size layers
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if v.numel() == model_dict[k].numel()
    }
    # 3. skip reinit layers
    if c.train_config.reinit_layers is not None:
        for reinit_layer_name in c.train_config.reinit_layers:
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if reinit_layer_name not in k
            }
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict),
                                                     len(model_dict)))
    return model_dict