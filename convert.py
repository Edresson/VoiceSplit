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
from utils.generic_utils import mix_wavfiles
from utils.generic_utils import load_config
import pandas as  pd

import librosa

config = load_config('config.json')
ap = AudioProcessor(config.audio)

data_path = '../test-my-data-prepo/train/'
files = os.listdir(data_path)
for file_name in files:
    if '.pt' in file_name:
       spec = ap.inv_spectrogram(torch.load(os.path.join(data_path,file_name)).cpu().detach().numpy())
       ap.save_wav(spec, os.path.join(data_path,file_name+'.wav'))
