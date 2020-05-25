import os
import time
import logging
import argparse

from utils.generic_utils import load_config
from utils.tensorboard import TensorboardWriter

from utils.train import train

from utils.dataset import train_dataloader, test_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset_dir', type=str, default='./',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('-m', '--model', type=str, default='voicefilter',
                        help="Name of the model. Used for model choise and for both logging and saving checkpoints. Valids values 'voicefilter' and voiceSplit")
    args = parser.parse_args()

    c = load_config(args.config_path)

    tensorboard = TensorboardWriter(log_dir, c)

    log_path = os.path.join(c.logs_path, args.model)
    os.makedirs(log_path, exist_ok=True)
   
    if(not os.path.isdir(c.dataset.train_dir)) or (not os.path.isdir(c.dataset.test_dir)):
        raise Exception("Please verify directories of dataset in "+args.config_path)

    train_dataloader = train_dataloader(c)
    test_dataloader = test_dataloader(c)
    
    #train(args, log_path, args.checkpoint_path, trainloader, testloader, tensorboard, c)