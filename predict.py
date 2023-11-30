import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time
from PIL import Image
import argparse
import os.path as osp
import yaml
import os.path as osp
import os
from utils import get_obj_from_string
import importlib
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
parser.add_argument('--exp', type=str, default=0,help="path experiment config")
parser.add_argument('--ckpt', type=str, default="",help="path to checkpoint")
parser.add_argument('--input', type=str, default="",help="path to dataset")
parser.add_argument('--output', type=str, default="",help="path to save results")
parser.add_argument('--resize', type=int, default=0,help="resize input images")
args = parser.parse_args()

with open(args.exp,'r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


device = torch.device(f'cuda:{args.cuda_id}')
config['model_name'] = config['model_file'].split('.')[1]

if len(args.wild.split('/')[-1]) != 0:
    wild_name = args.wild.split('/')[-1]
else:
    wild_name = args.wild.split('/')[-2]
val_vis1 = f"./{args.output}/cc_imgs"
val_vis2 = f"./{args.output}/final_imgs"
if not osp.exists(val_vis1):
    os.makedirs(val_vis1)

if not osp.exists(val_vis2):
    os.makedirs(val_vis2)

def data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size))
    return image,label

def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size))
    return image,label


def get_file_lst(file_path,gt_path):

  
    file_list = os.listdir(file_path)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        gt_f_path = os.path.join(gt_path,f)
        path_lst.append((f_path,gt_f_path))
    train_lst = path_lst
    return train_lst

def get_file_lst_c60(file_path):
 
    file_list = os.listdir(file_path)
    path_lst = []
    for f in file_list:
        f_path = os.path.join(file_path,f)
        path_lst.append((f_path,))
    train_lst = path_lst
    return train_lst


class UIE_c60(Dataset):
    def __init__(self,train_lst,image_size):
        self.path_lst =train_lst
        self.image_size = image_size
        print(f"got {len(self.path_lst)} images,{len(self.path_lst)} references")
    def __getitem__(self,index):
        name = self.path_lst[index][0].split('/')[-1]
        img = Image.open(self.path_lst[index][0]).convert('RGB')       
        image = TF.to_tensor(img)       
        image,ref = val_data_aug(image,image,self.image_size)
        
        return {
            'image': image,
            'name':name
        }
    def __len__(self):
        return len(self.path_lst)

val_lst = get_file_lst_c60(args.input)
val_set = UIE_c60(val_lst,image_size=None)

val_loader = DataLoader(val_set,batch_size = 1,shuffle=False,num_workers=8,pin_memory=False)


in_channels = 3
num_class = 3

net = get_obj_from_string(config['model_file'])(in_channels,num_class,img_size=config['img_size']).to(device)


def predict(net,args):
    ckpt = torch.load(f'args.ckpt',map_location=device)
    net.load_state_dict(ckpt['net'])
    print(f'Loading Done!')  
    net.eval()
    print(f'start inference...')
    with torch.no_grad():
        for data in tqdm(val_loader):
            name = data['name']
            if args.resize == 0:      
                image = data['image'].to(device)
            else:
                size = args.resize
                image = data['image']
                b,c,h,w = image.shape
                image = F.interpolate(image,size=(size,size),mode='bilinear').to(device)

            cc_pred,pred = net(image)
            if args.resize != 0:
                pred = F.interpolate(pred,size=(h,w),mode='bilinear')
                cc_pred = F.interpolate(cc_pred,size=(h,w),mode='bilinear')
            torchvision.utils.save_image(cc_pred,f'{val_vis1}/{name[0][:-4]}.png')
            torchvision.utils.save_image(pred,f'{val_vis2}/{name[0][:-4]}.png')
predict(net,args)