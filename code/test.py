from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D
import os
from torch import nn, optim
from model import get_model, train_model, test_model, test_model_tta
import yaml
from utils import load_config, save_checkpoint, map_fun
from dataset import IFLYTEK_Dataset
import argparse
from tqdm import tqdm
from trans import augmentation, trans
import torch
import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    init_parser = argparse.ArgumentParser(description='IPYFLY Baseline')
    init_parser.add_argument('--config_path', type=str, help='train config')
    init_args, init_extras = init_parser.parse_known_args()
    
    cfg = load_config(init_args.config_path)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpuid']
    
    weights_path_list = os.listdir(cfg['ckpt_root'])
    
    test_dataset = IFLYTEK_Dataset(cfg['test_folder'], aug=augmentation, trans=trans, denoise_list=cfg['denoise_list'], phase='test', tta=cfg['tta'])
    test_loader = D.DataLoader(
            test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_worker'])
    
    y_test_pre = np.zeros((len(test_dataset),cfg['class_num']))
    for i in range(cfg['n_fold']):
        print('[INFO] Start Inference Fold {}'.format(i))
        model = get_model(cfg)
        ckpt = torch.load(os.path.join(cfg['ckpt_root'],weights_path_list[i]), map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model = model.cuda()
        
        if cfg['tta']:
            print('[INFO] Using TTA')
            y_pre_temp = test_model_tta(test_loader, model, cfg) 
        else:
            y_pre_temp = test_model(test_loader, model, cfg) 

        y_test_pre += y_pre_temp / cfg['n_fold']
        
    y_pre = np.argmax(y_test_pre,axis=1)
    res = pd.DataFrame()
    res['image_id'] = test_dataset.get_image_id_list()
    res['image_id'] = res['image_id'].astype('int32')
    res['category_id'] = map_fun(y_pre)
    res.to_csv(cfg['submit_csv'],index=False)
    print('[INFO] Finish Inference')