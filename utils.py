import torch
import os
import numpy as np 
from PIL import Image
from scipy.stats import pearsonr
import importlib
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torch.nn as nn
import torchvision.models as models
import skimage

def data_aug(image,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
    return image

def data_crop(image,crop=None):
    if crop:
        height,width = image.shape[1], image.shape[2]
        top = max(0,(height - crop)//2)
        left = max(0,(width - crop)//2)
        image = TF.crop(image,top,left, crop, crop)
    return image


def val_data_aug(image,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
    return image

class UIE_train(Dataset):
    def __init__(self,train_lst,image_size):
        #file_list = os.listdir(file_path)
        self.path_lst =train_lst
        self.image_size = image_size
#         image_list = []
#         label_list = []
        #print("reading dataset.....")
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")
       # print(f'for check:{self.path_lst[10]}')
    def __getitem__(self,index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0])
        ref = Image.open(self.path_lst[index][1])
        if len(self.path_lst[index]) == 2:
            tx = img
        else:
            tx = Image.open(self.path_lst[index][2])
        
        image = TF.to_tensor(img)#.float().contiguous()
        ref = TF.to_tensor(ref)#.long().contiguous()#to tensor 会直接在最外层加一个维度
        tx = TF.to_tensor(tx)
        
        image,ref = data_aug(image,self.image_size),data_aug(ref,self.image_size)
        tx = data_aug(tx,self.image_size)
        
        return {
            'image': image,#unsqueeze add channel dim
            'ref': ref,
            'tx' : tx,
            'name':name

        }
    def __len__(self):
        return len(self.path_lst)
    

class UIE_val(Dataset):
    def __init__(self,train_lst,image_size):
        #file_list = os.listdir(file_path)
        self.path_lst =train_lst
        self.image_size = image_size
#         image_list = []
#         label_list = []
        #print("reading dataset.....")
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")
    def __getitem__(self,index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0])
        ref = Image.open(self.path_lst[index][1])
        
        image = TF.to_tensor(img)#.float().contiguous()
        ref = TF.to_tensor(ref)#.long().contiguous()#to tensor 会直接在最外层加一个维度
        
        image,ref = data_aug(image,self.image_size),data_aug(ref,self.image_size)
        
        return {
            'image': image,#unsqueeze add channel dim
            'ref': ref,
            'name':name
        }
    def __len__(self):
        return len(self.path_lst)

def get_file_lst(file_path,gt_path):
    
    file_list = os.listdir(file_path)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        gt_f_path = os.path.join(gt_path,f)
        path_lst.append((f_path,gt_f_path))
    train_lst = path_lst
    return train_lst


def get_obj_from_string(string):
    module,clas = string.rsplit('.',1)
    return getattr(importlib.import_module(module,package=None),clas)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count