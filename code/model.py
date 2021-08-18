from efficientnet_pytorch import EfficientNet
import torchvision
import yaml
from utils import load_config, save_checkpoint
from sklearn.metrics import accuracy_score
import torch
import torchvision
import os
from torch import nn, optim 
import numpy as np

def get_model(cfg):
    
    if 'efficientnet' in cfg['model_name']:
        if cfg['pretrained_weight']!='None':
            print('[INFO] Load Local Pretrained Weight From: {}'.format(cfg['pretrained_weight']))
            model = EfficientNet.from_name(cfg['model_name'])
            model.load_state_dict(torch.load(cfg['pretrained_weight']))
        else:
            model = EfficientNet.from_pretrained(cfg['model_name'])
        
        in_features = model._fc.in_features
        model._fc = nn.Linear(in_features, cfg['class_num'])
    else:
        if cfg['pretrained_weight']!='None':
            print('[INFO] Load Local Pretrained Weight From: {}'.format(cfg['pretrained_weight']))
            model = eval('torchvision.models.{}(pretrained=False)'.format((cfg['model_name'])))
            model.load_state_dict(torch.load(cfg['pretrained_weight']))
        else:
            model = eval('torchvision.models.{}(pretrained=True)'.format((cfg['model_name'])))
    
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg['class_num'])
        
    return model

def train_model(train_loader, valid_loader, model, fold_num, cfg):
    
    os.makedirs(cfg['ckpt_root'], exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1
    print('[INFO] Start Training!!!')
    for epoch in range(cfg['n_epoch']):
        loss_record = []
        acc_record = []
        ###################
        # train the model #
        ###################
        model.train()
        for count, (image, labels) in enumerate(train_loader):
            image = image.cuda()
            target = labels.cuda()
            labels = labels.numpy()
            
            pred = model(image)
            
            loss = criterion(pred, target)
            
            pred = pred.cpu().detach().numpy()
            pred = np.argmax(pred, axis=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_acc = accuracy_score(labels,pred)
            acc_record.append(iter_acc)
            loss = loss.item()
            loss_record.append(loss)
            
            if count and count % cfg['print_fre'] == 0:
                print("[INFO] T-Iter %d: loss=%.4f ,acc=%.4f" % (count, loss, iter_acc))
        print("[INFO] Epoch %d: loss=%.4f ,acc=%.4f" % (epoch+1, np.mean(loss_record), np.mean(acc_record)))
        scheduler.step()
        ###################
        # valid the model #
        ###################
        model.eval()
        acc_record = []
        with torch.no_grad():
            for count, (image, labels) in enumerate(valid_loader):
                image = image.cuda()
                target = labels.cuda()
                labels = labels.numpy()

                pred = model(image)


                pred = pred.cpu().detach().numpy()
                pred = np.argmax(pred, axis=1)

                iter_acc = accuracy_score(labels,pred)
                acc_record.append(iter_acc)
                
        curr_acc = np.mean(acc_record)
        print('[INFO] Valid Accuracy:{}'.format(curr_acc))
        if curr_acc > best_acc:
            best_acc = curr_acc
            ckpt_path = os.path.join(cfg['ckpt_root'],'fold_{}.pth'.format(fold_num))
            save_checkpoint(ckpt_path, model.state_dict(), epoch, best_acc)
            print('[INFO] Best Model With Best Accuracy:{}'.format(best_acc))
    return best_acc

def test_model(test_loader, model, cfg):
    model.eval()
    y_test_pre = np.zeros((test_loader.dataset.__len__(),cfg['class_num']))
    with torch.no_grad():
        for count, (images) in enumerate(test_loader):
            images = images.cuda()
            y_test_pre[count*cfg['batch_size']:(count+1)*cfg['batch_size'],:] = model(images).cpu().numpy()
    return y_test_pre

def test_model_tta(test_loader, model, cfg):
    model.eval()
    y_test_pre = np.zeros((test_loader.dataset.__len__(),cfg['class_num']))
    with torch.no_grad():
        for count, (image1,image2) in enumerate(test_loader):
            image1 = image1.cuda()
            image2 = image2.cuda()
            y_test_pre[count*cfg['batch_size']:(count+1)*cfg['batch_size'],:] = (model(image1).cpu().numpy() + model(image2).cpu().numpy()) /2
    return y_test_pre