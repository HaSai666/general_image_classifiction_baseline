import json
import os
import pickle
import cv2

import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils import random_denoise, saltpepper
from trans import augmentation, trans

class IFLYTEK_Dataset(Dataset):
    def __init__(self,train_folder, trans=None, aug=None, phase='train', size=(224,224), 
                 denoise_list = ['median_bulr_3'], debug_mode=False, tta=False):
        self.train_folder = train_folder
        self.trans = trans
        self.aug = aug
        self.phase = phase
        self.size = size
        self.denoise_list = denoise_list
        self.debug_mode = debug_mode
        self.tta = tta
        self.gen_image_list()
        
    def gen_image_list(self):
        '''
        AD:1,NC:0
        '''
        self.image_list = []
        self.label_list = []
        self.image_id_list = []
        print('[INFO] Loading Dataset!!!')
        for root, dirs, files in os.walk(self.train_folder):
            for f in tqdm(files):
                if root.split('/')[-1] == 'AD':
                    self.image_list.append(os.path.join(root, f))
                    self.image_id_list.append(f.split('.')[0])
                    self.label_list.append(1)
                else:
                    self.image_list.append(os.path.join(root, f))
                    self.image_id_list.append(f.split('.')[0])
                    self.label_list.append(0)
        if self.debug_mode:
            self.image_list = self.image_list[:100]
            self.label_list = self.label_list[:100]
                    
    def __len__(self):
        return len(self.image_list)

    def get_image_id_list(self):
        return self.image_id_list
    
    def __getitem__(self,index):
        
        image = cv2.imread(self.image_list[index])
        label = self.label_list[index]
        
        if self.phase == 'train':
            image = random_denoise(image, self.denoise_list)
            image = cv2.resize(image, self.size)
            image = self.aug(image=np.array(image))['image']
            image = self.trans(np.array(image))
        elif self.tta:
            noise_image = saltpepper(image)
            image = cv2.resize(image, self.size)
            noise_image = cv2.resize(noise_image, self.size)
            image = self.trans(np.array(image))
            noise_image = self.trans(np.array(noise_image))
        else:
            image = cv2.resize(image, self.size)
            image = self.trans(np.array(image))
           
        if self.phase == 'train':
            return image, label
        elif self.tta:
            return image, noise_image
        else:
            return image