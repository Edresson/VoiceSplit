#!/usr/bin/env python
# coding: utf-8

# This is a noteboook used to generate the speaker embeddings with the  GE2E model.

# In[1]:


import sys 
sys.path.insert(0, "../")


# In[2]:


from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 
from utils.generic_utils import load_config
import librosa
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm


# In[3]:


#Download encoder Checkpoint
#!wget https://github.com/Edresson/GE2E-Speaker-Encoder/releases/download/checkpoints/checkpoint-voicefilter-seungwonpark.pt -O embedder.pt


# In[4]:


# speaker_encoder parameters 
num_mels = 40
n_fft = 512
emb_dim = 256
lstm_hidden = 768
lstm_layers = 3
window = 80
stride = 40

checkpoint_dir = "embedder.pt"


# In[5]:


import torch
import torch.nn as nn

class LinearNorm(nn.Module):
    def __init__(self, lstm_hidden, emb_dim):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(lstm_hidden, emb_dim)

    def forward(self, x):
        return self.linear_layer(x)


class SpeakerEncoder(nn.Module):
    def __init__(self, num_mels, lstm_layers, lstm_hidden, window, stride):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(num_mels, lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.proj = LinearNorm(lstm_hidden, emb_dim)
        self.num_mels = num_mels
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.window = window
        self.stride = stride

    def forward(self, mel):
        # (num_mels, T)
        mels = mel.unfold(1, self.window, self.stride) # (num_mels, T', window)
        mels = mels.permute(1, 2, 0) # (T', window, num_mels)
        x, _ = self.lstm(mels) # (T', window, lstm_hidden)
        x = x[:, -1, :] # (T', lstm_hidden), use last frame only
        x = self.proj(x) # (T', emb_dim)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True) # (T', emb_dim)
        x = x.sum(0) / x.size(0) # (emb_dim), average pooling over time frames
        return x


# In[6]:


embedder = SpeakerEncoder(num_mels, lstm_layers, lstm_hidden, window, stride).cuda()
chkpt_embed = torch.load(checkpoint_dir)
embedder.load_state_dict(chkpt_embed)
embedder.eval()


# In[7]:


# Set constants
DATA_ROOT_PATH = '../../../LibriSpeech/voicefilter-open-fiel-ao-paper-data/'
TRAIN_DATA = os.path.join(DATA_ROOT_PATH, 'train')
TEST_DATA = os.path.join(DATA_ROOT_PATH, 'test')
glob_re_wav_emb = '*-ref_emb.wav'
glob_re_emb = '*-emb.pt'


# In[8]:


# load ap compativel with speaker encoder
config = {"backend":"voicefilter", "mel_spec": False,  "audio_len": 3, 
          "voicefilter":{"n_fft": 1200,"num_mels":40,"num_freq": 601,"sample_rate": 16000,"hop_length": 160,
                         "win_length": 400,"min_level_db": -100.0, "ref_level_db": 20.0, "preemphasis": 0.97,
                         "power": 1.5, "griffin_lim_iters": 60}}
ap = AudioProcessor(config)


# In[9]:


os.listdir(TEST_DATA)


# In[12]:


#Preprocess dataset
train_files = sorted(glob(os.path.join(TRAIN_DATA, glob_re_wav_emb)))
test_files = sorted(glob(os.path.join(TEST_DATA, glob_re_wav_emb)))

if len(train_files) == 0 or len(test_files):
    print("check train and test path files not in directory")
files  = train_files+test_files

for i in tqdm(range(len(files))):
    try:
        wave_file_path = files[i]
        wav_file_name = os.path.basename(wave_file_path)
        # Extract Embedding
        with open(wave_file_path, 'r') as f:
            LB_wave_file_path = f.readline().strip()
        emb_wav, _ = librosa.load(os.path.join('../',LB_wave_file_path), sr=16000)
        mel = torch.from_numpy(ap.get_mel(emb_wav)).cuda()
        #print(mel.shape)
        file_embedding = embedder(mel).cpu().detach().numpy()
    except:
        # if is not possible extract embedding because wav lenght is very small
        file_embedding = np.array([0]) # its make a error in training
        print("Embedding reference is very sort")
    output_name = wave_file_path.replace(glob_re_wav_emb.replace('*',''),'')+glob_re_emb.replace('*','')
    torch.save(torch.from_numpy(file_embedding.reshape(-1)), output_name)


# In[14]:


#file_embedding


# In[ ]:




