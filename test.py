import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, load_config_from_str
from utils.generic_utils import set_init_dict

from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

from utils.generic_utils import validation, PowerLaw_Compressed_Loss, SiSNR_With_Pit

from models.voicefilter.model import VoiceFilter
from models.voicesplit.model import VoiceSplit
from utils.audio_processor import WrapperAudioProcessor as AudioProcessor 

def test(args, log_dir, checkpoint_path, testloader, tensorboard, c, model_name, ap, cuda=True):
    if(model_name == 'voicefilter'):
        model = VoiceFilter(c)
    elif(model_name == 'voicesplit'):
        model = VoiceSplit(c)
    else:
        raise Exception(" The model '"+model_name+"' is not suported")

    if c.train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=c.train_config['learning_rate'])
    else:
        raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])

    step = 0
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if cuda:
                model = model.cuda()
        except:
            raise Exception("Fail in load checkpoint, you need use this configs: %s" %checkpoint['config_str'])
        
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print(" > Optimizer state is not loaded from checkpoint path, you see this mybe you change the optimizer")
        
        step = checkpoint['step']
    else:
        raise Exception("You need specific a checkpoint for test")
    # convert model from cuda
    if cuda:
        model = model.cuda()

    # definitions for power-law compressed loss
    power = c.loss['power']
    complex_ratio = c.loss['complex_loss_ratio']

    if c.loss['loss_name'] == 'power_law_compression':
        criterion = PowerLaw_Compressed_Loss(power, complex_ratio)
    elif c.loss['loss_name'] == 'si_snr':
        criterion = SiSNR_With_Pit()
    else:
        raise Exception(" The loss '"+c.loss['loss_name']+"' is not suported")
    return validation(criterion, ap, model, testloader, tensorboard, step,  cuda=cuda, loss_name=c.loss['loss_name'], test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str, default='./',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config_path', type=str, required=False, default=None,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file, for continue training")
    args = parser.parse_args()

    if args.config_path:
        c = load_config(args.config_path)
    else: #load config in checkpoint
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        c = load_config_from_str(checkpoint['config_str'])

    ap = AudioProcessor(c.audio)

    log_path = os.path.join(c.train_config['logs_path'], c.model_name)
    audio_config = c.audio[c.audio['backend']]
    tensorboard = TensorboardWriter(log_path, audio_config)
    # set test dataset dir
    c.dataset['test_dir'] = args.dataset_dir

    test_dataloader = test_dataloader(c, ap)
    mean_loss, mean_sdr = test(args, log_path, args.checkpoint_path, test_dataloader, tensorboard, c, c.model_name, ap, cuda=True)