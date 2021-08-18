from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import os
from torch import nn, optim
from model import get_model, train_model
import yaml
from utils import load_config, save_checkpoint
from dataset import IFLYTEK_Dataset
import argparse
from tqdm import tqdm
from trans import augmentation, trans
import torch
import os
import time
import numpy as np

if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='IPYFLY Baseline')
    init_parser.add_argument('--config_path', type=str, help='train config')
    init_args, init_extras = init_parser.parse_known_args()
    
    cfg = load_config(init_args.config_path)
    
    cfg['ckpt_root'] = cfg['ckpt_root'] + '_'+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpuid']
    
    dataset = IFLYTEK_Dataset(cfg['train_folder'], aug=augmentation, trans=trans, denoise_list=cfg['denoise_list'], debug_mode=cfg['debug_mode'])
    
    kf=KFold(n_splits=cfg['n_fold'],shuffle=True)
    fold_num = 0
    fold_acc_list = []
    for train_index,valid_index in kf.split(range(len(dataset))):

        train_dataset = D.Subset(dataset, train_index)
        valid_dataset = D.Subset(dataset, valid_index)

        train_loader = D.DataLoader(
            train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_worker'])

        valid_loader = D.DataLoader(
            valid_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_worker'])

        model = get_model(cfg)
        model = model.cuda()

        fold_acc = train_model(train_loader, valid_loader, model, fold_num, cfg)
        fold_acc_list.append(fold_acc)

        fold_num +=1
    for i in range(cfg['n_fold']):
        print('Fold {} Accuracy:{}'.format(i,fold_acc_list[i]))
    print('[INFO] Finish Training Local CV:{}'.format(np.mean(fold_acc_list)))