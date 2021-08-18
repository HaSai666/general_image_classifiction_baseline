import cv2 as cv
import random
import yaml
import torch
import numpy as np

def saltpepper(img,n=0.02):
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=255
        elif img.ndim==3:
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img

def median_bulr_3(image):
    medianbulr = cv.medianBlur(image,3)
    return medianbulr

def median_bulr_5(image):
    medianbulr = cv.medianBlur(image,5)
    return medianbulr

def median_bulr_7(image):
    medianbulr = cv.medianBlur(image,7)
    return medianbulr

def random_denoise(image, denoise_list = ['median_bulr_3','median_bulr_5','median_bulr_7','origin']):
    denoise_fun = random.choice(denoise_list)
    if denoise_fun != 'origin':
        return eval(denoise_fun)(image)
    else:
        return image
    
def map_fun(y_pre,map_dict={0:'NC',1:'AD'}):
    cate_list = []
    for y in (y_pre):
        cate_list.append(map_dict[y])
    return cate_list
    
def load_config(config_file):
    f = open(config_file,'r',encoding='utf-8')
    a = f.read()
    result = yaml.load(a,Loader=yaml.FullLoader)
    return result

def save_checkpoint(path, state_dict, epoch=0, acc=0):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if torch.is_tensor(v):
            v = v.cpu()
        new_state_dict[k] = v

    torch.save({
        "epoch": epoch,
        "acc": acc,
        "state_dict": new_state_dict,
    }, path)