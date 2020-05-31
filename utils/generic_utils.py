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

def mix_wavfiles(output_dir, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path):
    data_out_dir = output_dir
    emb_audio, _ = librosa.load(embedding_utterance_path, sr=sample_rate)
    clean_audio, _ = librosa.load(clean_utterance_path, sr=sample_rate)
    interference, _ = librosa.load(interference_utterance_path, sr=sample_rate)
    assert len(d.shape) == len(clean_audio.shape) == len(interference.shape) == 1, \
        'wav files must be mono, not stereo'

    # trim initial and end  wave file silence using librosa
    emb_audio, _ = librosa.effects.trim(emb_audio, top_db=20)
    clean_audio, _ = librosa.effects.trim(clean_audio, top_db=20)
    interference, _ = librosa.effects.trim(interference, top_db=20)

    # calculate frames using audio necessary for config.audio['audio_len'] seconds
    audio_len_seconds = int(sample_rate * hp.data.audio_len)

    # if merged audio is shorter than audio_len_seconds, discard it
    if clean_audio.shape[0] < audio_len_seconds or interference.shape[0] < audio_len_seconds:
        return

    # merge audio
    mixed_audio = clean_audio[:audio_len_seconds] + interference[:audio_len_seconds]

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
    clean_spec = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)

def glob_re_to_filename(dire, glob, num):
    return os.path.join(dire, glob.replace('*', '%06d' % num))

def powerlaw_compressed_loss(criterion, output, interference, power, complex_loss_ratio):
    # criterion is nn.MSELoss() instance
    # Power-law compressed loss
    spec_loss = criterion(torch.pow(torch.abs(output), power), torch.pow(torch.abs(interference), power))
    complex_loss = criterion(torch.pow(torch.clamp(output, min=0.0), power), torch.pow(torch.clamp(interference, min=0.0), power))
    loss = spec_loss + (complex_loss * complex_loss_ratio)

    return loss

def validation(criterion, ap, model, embedder, testloader, writer, step, cuda=True):
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            emb, clean_spec, mixed_spec, interference_wav, mixed_wav = batch[0]

            emb = emb.unsqueeze(0)
            clean_spec = clean_spec.unsqueeze(0)
            mixed_spec = mixed_spec.unsqueeze(0)

            if cuda:
                emb = emb.cuda()
                clean_spec = clean_spec.cuda()
                mixed_spec = mixed_spec.cuda()
        
            est_mask = model(mixed_spec, emb)
            est_mag = est_mask * mixed_spec
            test_loss = criterion(clean_spec, est_mag).item()

            mixed_spec = mixed_spec[0].cpu().detach().numpy()
            clean_spec = clean_spec[0].cpu().detach().numpy()
            est_mag = est_mag[0].cpu().detach().numpy()
            est_wav = ap.inv_spectrogram(est_mag)
            est_mask = est_mask[0].cpu().detach().numpy()

            sdr = bss_eval_sources(interference_wav, est_wav, False)[0][0]
            writer.log_evaluation(test_loss, sdr,
                                  mixed_wav, interference_wav, est_wav,
                                  mixed_spec.T, clean_spec.T, est_mag.T, est_mask.T,
                                  step)
            break

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