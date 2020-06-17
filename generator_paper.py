import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio import Audio
from utils.hparams import HParam
import pandas as pd
def formatter(dir_, form, num):
    return os.path.join(dir_, form.replace('*', '%06d' % num))

def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def mix(out_dir,hp, args, audio, num, s1_dvec, s1_target, s2):
    dir_ = out_dir
    srate = hp.audio.sample_rate

    d, _ = librosa.load(s1_dvec, sr=srate)
    w1, _ = librosa.load(s1_target, sr=srate)
    w2, _ = librosa.load(s2, sr=srate)
    assert len(d.shape) == len(w1.shape) == len(w2.shape) == 1, \
        'wav files must be mono, not stereo'

    d, _ = librosa.effects.trim(d, top_db=20)
    w1, _ = librosa.effects.trim(w1, top_db=20)
    w2, _ = librosa.effects.trim(w2, top_db=20)

    # if reference for d-vector is too short, discard it
    if d.shape[0] < 1.1 * hp.embedder.window * hp.audio.hop_length:
        return

    # I think random segment length will be better, but let's follow the paper first
    # fit audio to `hp.data.audio_len` seconds.
    # if merged audio is shorter than `L`, discard it
    L = int(srate * hp.data.audio_len)
    if w1.shape[0] < L or w2.shape[0] < L:
        return
    w1, w2 = w1[:L], w2[:L]

    mixed = w1 + w2

    norm = np.max(np.abs(mixed)) * 1.1
    w1, w2, mixed = w1/norm, w2/norm, mixed/norm

    # save vad & normalized wav files
    target_wav_path = formatter(dir_, hp.form.target.wav, num)
    mixed_wav_path = formatter(dir_, hp.form.mixed.wav, num)
    librosa.output.write_wav(target_wav_path, w1, srate)
    librosa.output.write_wav(mixed_wav_path, mixed, srate)

    # save magnitude spectrograms
    target_mag, _ = audio.wav2spec(w1)
    mixed_mag, _ = audio.wav2spec(mixed)
    target_mag_path = formatter(dir_, hp.form.target.mag, num)
    mixed_mag_path = formatter(dir_, hp.form.mixed.mag, num)
    torch.save(torch.from_numpy(target_mag), target_mag_path)
    torch.save(torch.from_numpy(mixed_mag), mixed_mag_path)

    # save selected sample as text file. d-vec will be calculated soon
    dvec_text_path = formatter(dir_, hp.form.dvec, num)
    with open(dvec_text_path, 'w') as f:
        f.write(s1_dvec)


if __name__ == '__main__':
    def train_wrapper(num):
        clean_utterance_path, embedding_utterance_path, interference_utterance_path = train_data[num]
        mix(output_dir_train, hp, args, audio, num, embedding_utterance_path, clean_utterance_path, interference_utterance_path)

        #mix_wavfiles(output_dir_train, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path)
    def test_wrapper(num):
        clean_utterance_path, embedding_utterance_path, interference_utterance_path = test_data[num]
        mix(output_dir_test, hp, args, audio, num, embedding_utterance_path, clean_utterance_path, interference_utterance_path)
        #mix_wavfiles(output_dir_test, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Config json file")
    parser.add_argument('-r', '--dataset_root_dir', type=str, required=True,
                        help="Config yaml file")               
    parser.add_argument('-d', '--train_data_csv', type=str, required=True,
                        help="Train Data csv contains rows [clean_utterance,embedding_utterance,interference_utterance] example in datasets/LibriSpeech/train.csv")
    parser.add_argument('-t', '--test_data_csv', type=str, required=True,
                        help="Test Data csv contains rows [clean_utterance,embedding_utterance,interference_utterance] example in datasets/LibriSpeech/dev.csv")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-l', '--librispeech', type=str, required=False, default=False,
                        help="Librispeech format, if true load with librispeech format")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    cpu_num = cpu_count() # num threads = num cpu cores 

    hp = HParam(args.config)
    audio = Audio(hp)

    output_dir_train = os.path.join(args.out_dir, 'train')
    output_dir_test = os.path.join(args.out_dir, 'test')

    dataset_root_dir = args.dataset_root_dir


    train_data_csv = pd.read_csv(args.train_data_csv, sep=',').values
    test_data_csv = pd.read_csv(args.test_data_csv, sep=',').values
    train_data = []
    test_data = []
    if args.librispeech:
        for c, e, i in train_data_csv:
            splits = c.split('-')
            target_path = os.path.join(dataset_root_dir, splits[0], splits[1], c+'-norm.wav')
            splits = e.split('-')
            emb_ref_path = os.path.join(dataset_root_dir, splits[0], splits[1], e+'-norm.wav')
            splits = i.split('-')
            interference_path = os.path.join(dataset_root_dir, splits[0], splits[1], i+'-norm.wav')           
            train_data.append([target_path, emb_ref_path, interference_path])
        
        for c, e, i in test_data_csv:
            splits = c.split('-')
            target_path = os.path.join(dataset_root_dir, splits[0], splits[1], c+'-norm.wav')
            splits = e.split('-')
            emb_ref_path = os.path.join(dataset_root_dir, splits[0], splits[1], e+'-norm.wav')
            splits = i.split('-')
            interference_path = os.path.join(dataset_root_dir, splits[0], splits[1], i+'-norm.wav')           
            test_data.append([target_path, emb_ref_path, interference_path])
    else:
        for c, e, i in train_data_csv:
            train_data.append([os.path.join(dataset_root_dir,c), os.path.join(dataset_root_dir,e), os.path.join(dataset_root_dir,i)])
        
        for c, e, i in test_data_csv:
            test_data.append([os.path.join(dataset_root_dir,c), os.path.join(dataset_root_dir,e), os.path.join(dataset_root_dir,i)])
    
    train_idx = list(range(len(train_data)))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(train_wrapper, train_idx), total=len(train_idx)))

    test_idx = list(range(len(test_data)))
    with Pool(cpu_num) as p:
        r = list(tqdm.tqdm(p.imap(test_wrapper, test_idx), total=len(test_idx)))
