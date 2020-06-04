import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack

class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """
    def __init__(self, c, ap, train=True):
        self.c = c
        self.ap = ap
        self.train = train
        self.dataset_dir = c.dataset['train_dir'] if train else c.dataset['test_dir']
        assert os.path.isdir(self.dataset_dir),'Test or Train dataset dir is incorrect! Fix it in config.json'
        
        format_data = c.dataset['format']
        self.emb_list = self.find_files_by_format(format_data['emb'])
        self.target_spec_list = self.find_files_by_format(format_data['target'])
        self.mixed_spec_list = self.find_files_by_format(format_data['mixed'])
        if not train:
            self.target_wav_list = self.find_files_by_format(format_data['target_wav'])
            self.mixed_wav_list = self.find_files_by_format(format_data['mixed_wav'])
        # asserts for integrity
        assert len(self.emb_list) == len(self.target_spec_list) == len(self.mixed_spec_list), " The number of target and mixed Specs and Embs not Match! Check its"
        assert len(self.target_spec_list) != 0, " Training files not found !"

    def find_files_by_format(self, glob_exp):
            return sorted(glob(os.path.join(self.dataset_dir, glob_exp)))

    def __getitem__(self, idx):
        if self.train:
            return torch.load(self.emb_list[idx]), torch.load(self.target_spec_list[idx]), torch.load(self.mixed_spec_list[idx])      
        else: # if test
            emb = torch.load(self.emb_list[idx])
            target_spec = torch.load(self.target_spec_list[idx])
            mixed_spec = torch.load(self.mixed_spec_list[idx])
            mixed_wav = self.ap.load_wav(self.mixed_wav_list[idx])
            target_wav = self.ap.load_wav(self.target_wav_list[idx])
            return emb, target_spec, mixed_spec, target_wav, mixed_wav
    def __len__(self):
        return len(self.emb_list)

def train_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=True),
                          batch_size=c.train_config['batch_size'],
                          shuffle=True,
                          num_workers=c.train_config['num_workers'],
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)

def test_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=False),
                          collate_fn=test_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])

def train_collate_fn(item):
    embs_list = []
    target_list = []
    mixed_list = []
    for emb, target, mixed in item:
        embs_list.append(emb)
        target_list.append(target)
        mixed_list.append(mixed)
    # concate tensors in dim 0
    target_list = stack(target_list, dim=0)
    mixed_list = stack(mixed_list, dim=0)
    embs_list = stack(embs_list, dim=0)
    return embs_list, target_list, mixed_list

def test_collate_fn(batch):
        return batch