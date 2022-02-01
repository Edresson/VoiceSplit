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
import yaml
from sklearn.preprocessing import minmax_scale

import random
from random import getrandbits
# set random seed
random.seed(0)

def get_audios_with_random_amp(emb_audio, clean_audio, interference, noise_audio):
    # noise audio amp is small than mean between clean 1 and interference 2.
    # random amplitude sinal
    min_amp = random.uniform(-1, -0.3)
    max_amp = (min_amp*-1)+random.uniform(0.0, 0.02)
    shape = emb_audio.shape
    emb_audio = minmax_scale(emb_audio.ravel(), feature_range=(min_amp,max_amp)).reshape(shape)

    min_amp_clean = random.uniform(-1, -0.3)
    max_amp_clean = (min_amp_clean*-1)+random.uniform(0.0, 0.02)
    shape = clean_audio.shape
    clean_audio = minmax_scale(clean_audio.ravel(), feature_range=(min_amp_clean,max_amp_clean)).reshape(shape)

    min_amp_interference = random.uniform(-1, -0.3)
    max_amp_interference = (min_amp_interference*-1)+random.uniform(0.0, 0.02)
    shape = interference.shape
    interference = minmax_scale(interference.ravel(), feature_range=(min_amp_interference,max_amp_interference)).reshape(shape)
    
    min_noise = random.uniform(min(min_amp_clean,min_amp_interference),-0.1 )
    max_noise = (min_noise*-1)-random.uniform(0.0, 0.02) #random.uniform(0.3, min(max_amp_clean,max_amp_interference))
    #print(min_noise,max_noise, noise_audio.shape)
    shape = noise_audio.shape
    noise_audio = minmax_scale(noise_audio.ravel(), feature_range=(min_noise,max_noise)).reshape(shape)

    return emb_audio, clean_audio, interference, noise_audio

def mix_wavfiles_without_voice_overlay(output_dir, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path, noise_1_path, noise_2_path):
    data_out_dir = output_dir
    emb_audio, _ = librosa.load(embedding_utterance_path, sr=sample_rate)
    clean_audio, _ = librosa.load(clean_utterance_path, sr=sample_rate)
    interference, _ = librosa.load(interference_utterance_path, sr=sample_rate)
    noise_1_audio, _ = librosa.load(noise_1_path, sr=sample_rate)
    noise_2_audio, _ = librosa.load(noise_2_path, sr=sample_rate)

    assert len(emb_audio.shape) == len(clean_audio.shape) == len(interference.shape) == 1, \
        'wav files must be mono, not stereo'

    # trim initial and end  wave file silence using librosa
    emb_audio, _ = librosa.effects.trim(emb_audio, top_db=20)
    clean_audio, _ = librosa.effects.trim(clean_audio, top_db=20)
    interference, _ = librosa.effects.trim(interference, top_db=20)

    '''norm_factor = np.max(np.abs(clean_audio)) * 1.1
    interference = interference / norm_factor
    clean_audio = clean_audio / norm_factor'''

    # if reference for emebdding is too short, discard it
    window = 80
    hop_length = 160

    if emb_audio.shape[0] < 1.1*window*hop_length:
        return

    two_clean = not getrandbits(1)
    
    # calculate frames using audio necessary for config.audio['audio_len'] seconds
    seconds = random.randint(2,4)
    audio_len_clean_max_value = int(sample_rate * seconds)

    seconds = random.randint(2,4)
    audio_len_interference_max_value = int(sample_rate * seconds)

    #print(audio_len_interference_max_value, audio_len_clean_max_value,len(noise_1_audio))
    out_audio_len = audio_len_clean_max_value+audio_len_interference_max_value
    len_noise1 = len(noise_1_audio)
    len_noise2 = len(noise_2_audio)
    # calculate using smaller noise audio
    noise_start_slice = random.randint(0,min(len_noise1,len_noise2)-(out_audio_len+1))
    # sum two diferents noise
    noise_audio = noise_1_audio[noise_start_slice:noise_start_slice+out_audio_len]+noise_2_audio[noise_start_slice:noise_start_slice+out_audio_len]
    #print(noise_audio.shape)
    # if merged audio is shorter than audio_len_seconds, discard it
    if clean_audio.shape[0] < audio_len_clean_max_value or interference.shape[0] < audio_len_interference_max_value:
        return

    emb_audio_random, clean_audio_random, interference_random, noise_audio_random = get_audios_with_random_amp(emb_audio, clean_audio, interference, noise_audio)
    
    # normalise noise file
    min_noise = random.uniform(min(clean_audio.min(),interference.min()),-0.1)
    max_noise = (min_noise*-1)-random.uniform(0.0, 0.02) #random.uniform(0.3, min(max_amp_clean,max_amp_interference))

    #print(min_noise,max_noise, noise_audio.shape)
    shape = noise_audio.shape
    noise_audio = minmax_scale(noise_audio.ravel(), feature_range=(min_noise,max_noise)).reshape(shape)

    clean_audio = clean_audio[:audio_len_clean_max_value]
    interference = interference[:audio_len_interference_max_value]
    # random amp file 
    clean_audio_random = clean_audio_random[:audio_len_clean_max_value]
    interference_random = interference_random[:audio_len_interference_max_value]

    # preparar ruido para as partes ( o mesmo ruido ) ( concatenar 2 ruidos diferentes)
    
    
    
    if two_clean:
        clean_audio_parts = librosa.effects.split(clean_audio,top_db=20)
        # adicionar rudio em  interference, clean_audio,  interference_random, clean_audio_random         
        if len(clean_audio_parts) > 1:
            #pega a metade e concatena
            clip_idx = clean_audio_parts[int(len(clean_audio_parts)/2)][1]
            # generate audio with amp normalised
            part1 = clean_audio[:clip_idx]
            part2 = clean_audio[clip_idx:]
            len_part1 = len(part1)
            len_interference = len(interference)

            # noise without interruption
            part1 = part1 + noise_audio[:len_part1]
            interference = interference + noise_audio[len_part1:len_interference+len_part1]
            part2 = part2 + noise_audio[len_interference+len_part1:len_interference+len_part1+len(part2)]
            
            zeros_interference = np.zeros(interference.shape)
            mixed_audio = np.concatenate((part1, interference, part2))
            clean_audio_padded = np.concatenate((part1, zeros_interference, part2))

            # generate audios with random amp
            part1 = clean_audio_random[:clip_idx]
            part2 = clean_audio_random[clip_idx:]
            len_part1 = len(part1)
            len_interference = len(interference_random)
            # noise without interruption
            part1 = part1 + noise_audio_random[:len_part1]
            interference_random = interference_random + noise_audio_random[len_part1:len_interference+len_part1]
            part2 = part2 + noise_audio_random[len_interference+len_part1:len_interference+len_part1+len(part2)]
            zeros_interference = np.zeros(interference_random.shape)
            mixed_audio_random = np.concatenate((part1, interference_random, part2))
            clean_audio_padded_random = np.concatenate((part1, zeros_interference, part2))
        else:
            # adicionando ruido
            clean_audio = clean_audio + noise_audio[:len(clean_audio)]
            interference = interference + noise_audio[len(clean_audio):len(clean_audio)+len(interference)]
            zeros_interference = np.zeros(interference.shape)
            mixed_audio = np.concatenate((clean_audio, interference))
            clean_audio_padded = np.concatenate((clean_audio, zeros_interference))
            
            clean_audio_random = clean_audio_random + noise_audio_random[:len(clean_audio_random)]
            interference_random = interference_random + noise_audio_random[len(clean_audio_random):len(clean_audio_random)+len(interference_random)]
            mixed_audio_random = np.concatenate((clean_audio_random, interference_random))
            clean_audio_padded_random = np.concatenate((clean_audio_random, zeros_interference))

    else: 
        interference_parts = librosa.effects.split(interference,top_db=15)
        # adicionar rudio em  interference, clean_audio,  interference_random, clean_audio_random 
        if len(interference_parts) > 1:
            #pega a metade e concatena
            clip_idx = interference_parts[int(len(interference_parts)/2)][1]
            part1 = interference[:clip_idx]
            part2 = interference[clip_idx:]
            len_part1 = len(part1)
            len_clean = len(clean_audio)
            part1 = part1 + noise_audio[:len_part1]
            clean_audio = clean_audio + noise_audio[len_part1:len_clean+len_part1]
            part2 = part2 + noise_audio[len_clean+len_part1:len_clean+len_part1+len(part2)]

            zeros_part1 = np.zeros(part1.shape)
            zeros_part2 = np.zeros(part2.shape)
            mixed_audio = np.concatenate((part1, clean_audio, part2))
            clean_audio_padded = np.concatenate((zeros_part1, clean_audio, zeros_part2))
            
            part1 = interference_random[:clip_idx]
            part2 = interference_random[clip_idx:]

            len_part1 = len(part1)
            len_clean = len(clean_audio_random)
            # noise without interruption
            part1 = part1 + noise_audio_random[:len_part1]
            clean_audio_random = clean_audio_random + noise_audio_random[len_part1:len_clean+len_part1]
            part2 = part2 + noise_audio_random[len_clean+len_part1:len_clean+len_part1+len(part2)]


            zeros_part1 = np.zeros(part1.shape)
            zeros_part2 = np.zeros(part2.shape)
            mixed_audio_random = np.concatenate((part1, clean_audio_random, part2))
            clean_audio_padded_random = np.concatenate((zeros_part1, clean_audio_random, zeros_part2))
            

        else:
            interference = interference + noise_audio[:len(interference)]
            clean_audio = clean_audio + noise_audio[len(interference):len(clean_audio)+len(interference)]
            zeros_interference = np.zeros(interference.shape)
            mixed_audio = np.concatenate((interference, clean_audio))
            clean_audio_padded = np.concatenate((zeros_interference, clean_audio))

            interference_random = interference_random + noise_audio_random[:len(interference_random)]
            clean_audio_random = clean_audio_random + noise_audio[len(interference_random):len(clean_audio_random)+len(interference_random)]
            mixed_audio_random = np.concatenate((interference_random, clean_audio_random))
            clean_audio_padded_random = np.concatenate((zeros_interference, clean_audio_random))


    # normlise audio
    norm_factor = np.max(np.abs(mixed_audio)) * 1.1
    clean_audio_padded = clean_audio_padded/norm_factor
    emb_audio = emb_audio/norm_factor
    mixed_audio = mixed_audio/norm_factor

    interference = interference/norm_factor
    clean_audio = clean_audio/norm_factor

    interference_output = np.zeros(interference.shape)
    clean_audio_output = clean_audio

    norm_factor = np.max(np.abs(mixed_audio_random)) * 1.1
    clean_audio_padded_random = clean_audio_padded_random/norm_factor
    emb_audio_random = emb_audio_random/norm_factor
    mixed_audio_random = mixed_audio_random/norm_factor

    # salve normal files 
    target_wav_path = glob_re_to_filename(data_out_dir, form['target_wav'], num, sub=1)
    mixed_wav_path = glob_re_to_filename(data_out_dir, form['mixed_wav'], num, sub=1)
    emb_wav_path = glob_re_to_filename(data_out_dir, form['emb_wav'], num, sub=1)
    librosa.output.write_wav(emb_wav_path, emb_audio, sample_rate)
    librosa.output.write_wav(target_wav_path, clean_audio_padded, sample_rate)
    librosa.output.write_wav(mixed_wav_path, mixed_audio, sample_rate)

    # extract and save spectrograms
    clean_spec, _ = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec, _ = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num, sub=1)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num, sub=1)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)

    # save input = output
    target_wav_path = glob_re_to_filename(data_out_dir, form['target_wav'], num, sub=2)
    mixed_wav_path = glob_re_to_filename(data_out_dir, form['mixed_wav'], num, sub=2)
    emb_wav_path = glob_re_to_filename(data_out_dir, form['emb_wav'], num, sub=2)
    librosa.output.write_wav(emb_wav_path, emb_audio, sample_rate)
    librosa.output.write_wav(target_wav_path, clean_audio, sample_rate)
    librosa.output.write_wav(mixed_wav_path, clean_audio, sample_rate)

    # extract and save spectrograms
    clean_spec, _ = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec, _ = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num, sub=2)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num, sub=2)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)

    # save  output zero mask, the input dont have voice from embedding ref speaker
    target_wav_path = glob_re_to_filename(data_out_dir, form['target_wav'], num, sub=3)
    mixed_wav_path = glob_re_to_filename(data_out_dir, form['mixed_wav'], num, sub=3)
    emb_wav_path = glob_re_to_filename(data_out_dir, form['emb_wav'], num, sub=3)
    librosa.output.write_wav(emb_wav_path, emb_audio, sample_rate)
    librosa.output.write_wav(target_wav_path, interference_output, sample_rate)
    librosa.output.write_wav(mixed_wav_path, interference, sample_rate)

    # extract and save spectrograms
    clean_spec, _ = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec, _ = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num, sub=3)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num, sub=3)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)

    # save  random amplitude files
    target_wav_path = glob_re_to_filename(data_out_dir, form['target_wav'], num, sub=4)
    mixed_wav_path = glob_re_to_filename(data_out_dir, form['mixed_wav'], num, sub=4)
    emb_wav_path = glob_re_to_filename(data_out_dir, form['emb_wav'], num, sub=4)
    librosa.output.write_wav(emb_wav_path, emb_audio_random, sample_rate)
    librosa.output.write_wav(target_wav_path, clean_audio_padded_random, sample_rate)
    librosa.output.write_wav(mixed_wav_path, mixed_audio_random, sample_rate)

    # extract and save spectrograms
    clean_spec, _ = ap.get_spec_from_audio_path(target_wav_path) # we need to load the wav to maintain compatibility with all audio backend
    mixed_spec, _ = ap.get_spec_from_audio_path(mixed_wav_path)
    clean_spec_path = glob_re_to_filename(data_out_dir, form['target'], num, sub=4)
    mixed_spec_path = glob_re_to_filename(data_out_dir, form['mixed'], num, sub=4)
    torch.save(torch.from_numpy(clean_spec), clean_spec_path)
    torch.save(torch.from_numpy(mixed_spec), mixed_spec_path)



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

def glob_re_to_filename(dire, glob, num, sub=False):
    if sub:
        return os.path.join(dire, glob.replace('*', '%06d_%d' %(num,sub)))
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
    def forward(self, estimate_source, source, source_lengths):
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
    count = 0
    with torch.no_grad():
        for batch in testloader:
            try:
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
                mixed_phase = mixed_phase[0].cpu().detach().numpy()

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
                count+=1
                print(count, 'of',testloader.__len__())
            except:
                continue
        if test:
            mean_test_loss = np.array(losses).mean()
            mean_sdr = np.array(sdrs).mean()
            print("Mean Test Loss:", mean_test_loss)
            print("Mean Test SDR:", mean_sdr)
            return mean_test_loss, mean_sdr

def test_fast_with_si_srn(criterion, ap, model, testloader, tensorboard, step, cuda=True, loss_name='si_snr', test=False):
    losses = []
    model.eval()
    # set fast and best criterion
    criterion = SiSNR_With_Pit()
    count = 0
    with torch.no_grad():
        for emb, clean_spec, mixed_spec, clean_wav, mixed_wav, mixed_phase, seq_len in testloader:  
                if cuda:
                    emb = emb.cuda()
                    clean_spec = clean_spec.cuda()
                    mixed_spec = mixed_spec.cuda()
                    mixed_phase = mixed_phase.cuda()
                    seq_len = seq_len.cuda()
                est_mask = model(mixed_spec, emb)
                est_mag = est_mask * mixed_spec
                # convert spec to wav using phase
                output = ap.torch_inv_spectrogram(est_mag, mixed_phase)
                target = ap.torch_inv_spectrogram(clean_spec, mixed_phase)
                shape = list(target.shape)
                target = torch.reshape(target, [shape[0],1]+shape[1:]) # append channel dim
                output = torch.reshape(output, [shape[0],1]+shape[1:]) # append channel dim
                test_loss = criterion(output, target, seq_len).item()
                losses.append(test_loss)
           
        mean_test_loss = np.array(losses).mean()
        print("Mean Si-SRN with Pit Loss:", mean_test_loss)
        return mean_test_loss

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
    data = yaml.load(input_str, Loader=yaml.FullLoader)
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
