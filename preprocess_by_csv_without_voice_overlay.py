import os
import glob
import tqdm
import torch
import random
import librosa
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count

from utils.audio_processor import WrapperAudioProcessor as AudioProcessor
from utils.generic_utils import mix_wavfiles_without_voice_overlay
from utils.generic_utils import load_config
import pandas as  pd


if __name__ == '__main__':
    def train_wrapper(num):
        try:
            clean_utterance_path, embedding_utterance_path, interference_utterance_path, noise1, noise2 = train_data[num]
            mix_wavfiles_without_voice_overlay(output_dir_train, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path, noise1, noise2)
        except:
            print("Erro in sample: ", clean_utterance_path)
    def test_wrapper(num):
        try:
            clean_utterance_path, embedding_utterance_path, interference_utterance_path, noise1, noise2 = test_data[num]
            mix_wavfiles_without_voice_overlay(output_dir_test, sample_rate, audio_len, ap, form, num, embedding_utterance_path, interference_utterance_path, clean_utterance_path, noise1, noise2)
        except:
            print("Erro in sample: ", clean_utterance_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Config json file")
    parser.add_argument('-r', '--dataset_root_dir', type=str, required=True,
                        help="Data Root Dir")               
    parser.add_argument('-d', '--train_data_csv', type=str, required=False,default=False,
                        help="Train Data csv contains rows [clean_utterance,embedding_utterance,interference_utterance] example in datasets/LibriSpeech/train.csv")
    parser.add_argument('-t', '--test_data_csv', type=str, required=False,default=False,
                        help="Test Data csv contains rows [clean_utterance,embedding_utterance,interference_utterance] example in datasets/LibriSpeech/dev.csv")
    parser.add_argument('-n', '--noise_csv', type=str, required=True,
                        help="csv with noise files path")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="Directory of output training triplet")
    parser.add_argument('-l', '--librispeech', type=str, required=False, default=False,
                        help="Librispeech format, if true load with librispeech format")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.train_data_csv:
        os.makedirs(os.path.join(args.out_dir, 'train'), exist_ok=True)
    if args.test_data_csv:
        os.makedirs(os.path.join(args.out_dir, 'test'), exist_ok=True)

    cpu_num = cpu_count() # num threads = num cpu cores 

    config = load_config(args.config)
    ap = AudioProcessor(config.audio)

    sample_rate = config.audio[config.audio['backend']]['sample_rate']
    audio_len = config.audio['audio_len']
    form = config.dataset['format']
    output_dir_train = os.path.join(args.out_dir, 'train')
    output_dir_test = os.path.join(args.out_dir, 'test')

    dataset_root_dir = args.dataset_root_dir
    train_data_csv = None
    test_data_csv = None

    noise_files = open(args.noise_csv).readlines()

    if args.train_data_csv:
        train_data_csv = pd.read_csv(args.train_data_csv, sep=',').values
    if args.test_data_csv:
        test_data_csv = pd.read_csv(args.test_data_csv, sep=',').values

    train_data = []
    test_data = []
    if args.librispeech:
        if train_data_csv is not None:
            for c, e, i in train_data_csv:
                splits = c.split('-')
                target_path = os.path.join(dataset_root_dir, splits[0], splits[1], c+'-norm.wav')
                splits = e.split('-')
                emb_ref_path = os.path.join(dataset_root_dir, splits[0], splits[1], e+'-norm.wav')
                splits = i.split('-')
                interference_path = os.path.join(dataset_root_dir, splits[0], splits[1], i+'-norm.wav')           
                num_noise_files = len(noise_files)
                noise_files1 = os.path.join(dataset_root_dir, noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n','').replace('\n',''))
                noise_files2 = os.path.join(dataset_root_dir,noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                train_data.append([target_path, emb_ref_path, interference_path, noise_files1, noise_files2])
        if test_data_csv is not None:
            for c, e, i in test_data_csv:
                splits = c.split('-')
                target_path = os.path.join(dataset_root_dir, splits[0], splits[1], c+'-norm.wav')
                splits = e.split('-')
                emb_ref_path = os.path.join(dataset_root_dir, splits[0], splits[1], e+'-norm.wav')
                splits = i.split('-')
                interference_path = os.path.join(dataset_root_dir, splits[0], splits[1], i+'-norm.wav')           
                num_noise_files = len(noise_files)
                noise_files1 = os.path.join(dataset_root_dir, noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                noise_files2 = os.path.join(dataset_root_dir,noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                test_data.append([target_path, emb_ref_path, interference_path, noise_files1, noise_files2])
    else:
        if train_data_csv is not None:
            for c, e, i in train_data_csv:
                num_noise_files = len(noise_files)
                noise_files1 = os.path.join(dataset_root_dir, noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                noise_files2 = os.path.join(dataset_root_dir,noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                train_data.append([os.path.join(dataset_root_dir,c), os.path.join(dataset_root_dir,e), os.path.join(dataset_root_dir,i), noise_files1, noise_files2])
        if test_data_csv is not None:
            for c, e, i in test_data_csv:
                num_noise_files = len(noise_files)
                noise_files1 = os.path.join(dataset_root_dir, noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                noise_files2 = os.path.join(dataset_root_dir, noise_files[random.randint(0,num_noise_files)-1].replace('\n','').replace('\n',''))
                test_data.append([os.path.join(dataset_root_dir,c), os.path.join(dataset_root_dir,e), os.path.join(dataset_root_dir,i), noise_files1, noise_files2])

    if train_data_csv is not None:
        train_idx = list(range(len(train_data)))
        with Pool(cpu_num) as p:
            r = list(tqdm.tqdm(p.imap(train_wrapper, train_idx), total=len(train_idx)))
    if test_data_csv is not None:
        test_data  = test_data[:20]
        test_idx = list(range(len(test_data)))
        with Pool(cpu_num) as p:
            r = list(tqdm.tqdm(p.imap(test_wrapper, test_idx), total=len(test_idx)))
