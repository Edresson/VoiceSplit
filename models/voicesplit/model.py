"""
The own VoiceSplit model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.generic_utils import Mish

class VoiceSplit(nn.Module):
    def __init__(self, config):
        super(VoiceSplit, self).__init__()
        self.config = config
        self.audio = self.config.audio[self.config.audio['backend']]

        convs = [
            # cnn1
            nn.ZeroPad2d((3, 3, 0, 0)),
            nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
            nn.BatchNorm2d(64), Mish(),
            # cnn2
            nn.ZeroPad2d((0, 0, 3, 3)),
            nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
            nn.BatchNorm2d(64), Mish(),

            # cnn3
            nn.ZeroPad2d(2),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
            nn.BatchNorm2d(64), Mish(),

            # cnn4
            nn.ZeroPad2d((2, 2, 4, 4)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
            nn.BatchNorm2d(64), Mish(),

            # cnn5
            nn.ZeroPad2d((2, 2, 8, 8)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
            nn.BatchNorm2d(64), Mish(),

            # cnn6
            nn.ZeroPad2d((2, 2, 16, 16)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
            nn.BatchNorm2d(64), Mish(),

            # cnn7
            nn.ZeroPad2d((2, 2, 32, 32)),
            nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
            nn.BatchNorm2d(64), Mish(),

            # cnn8
            nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
            nn.BatchNorm2d(8), Mish()]

        self.conv = nn.Sequential(*convs)

        # ToDo: Change LSTM to QRNN
        self.lstm = nn.LSTM(
            8*self.audio['num_freq'] + self.config.model['emb_dim'],
            self.config.model['lstm_dim'],
            batch_first=True,
            bidirectional=True)
        self.fc1 = nn.Linear(2*self.config.model['lstm_dim'], self.config.model['fc1_dim'])
        
        self.fc2 = nn.Linear(self.config.model['fc1_dim'], self.config.model['fc2_dim'])

    def forward(self, x, speaker_embedding):
        # x: [B, T, num_freq]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_freq]
        x = self.conv(x)
        # x: [B, 8, T, num_freq]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, 8, num_freq]
        x = x.view(x.size(0), x.size(1), -1)
        # x: [B, T, 8*num_freq]
        # speaker_embedding: [B, emb_dim]
        speaker_embedding = speaker_embedding.unsqueeze(1)
        speaker_embedding = speaker_embedding.repeat(1, x.size(1), 1)
        # speaker_embedding: [B, T, emb_dim]

        x = torch.cat((x, speaker_embedding), dim=2) # [B, T, 8*num_freq + emb_dim]
        x, _ = self.lstm(x) # [B, T, 2*lstm_dim]
        x = F.relu(x)
        x = self.fc1(x) # x: [B, T, fc1_dim]
        x = F.relu(x)
        x = self.fc2(x) # x: [B, T, fc2_dim], fc2_dim == num_freq
        x = torch.sigmoid(x)
        
        return x
