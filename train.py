from logging import logProcesses
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
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
from utils import AverageMeter,get_obj_from_string,UIE_train,UIE_val,get_file_lst
import importlib
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid



parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
parser.add_argument('--exp', type=str, default=0,help="path experiment config")
parser.add_argument('--debug', type=bool, default=False,help="debug or not")
parser.add_argument('--resize_val', type=bool, default=False,help="resize validation or not")
args = parser.parse_args()

with open(args.exp,'r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

device = torch.device(f'cuda:{args.cuda_id}')
config['model_name'] = config['model_file'].split('.')[1]
weight_decay = config['weight_decay']

if args.debug:
    log_path = f"./debug/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_ccweight{config['cc_weight']}_log/"
    snapshot_path = log_path + 'weights/'# to save loss and dice score etc.

else:
    
    log_path = f"./exp_results/{config['name']}_{config['model_name']}_lr{config['lr']}_wd{config['weight_decay']}_epoch{config['num_epochs']}_ccweight{config['cc_weight']}_log/"# to save loss and dice score etc.
    snapshot_path = log_path + 'weights/'

if not osp.exists(snapshot_path):
    os.makedirs(snapshot_path)
if not osp.exists(log_path):
    os.makedirs(log_path)

f = open(log_path+'save_log.txt','w')
f.write(f'experiment config file:{args.exp}\n')
for k in config.keys():
    f.write(f'{k}: {config[k]}\n')

if 'UIEB' in config['name']:
    tr_input_path='../../UIE_new/DATASET/UIEB/Train/input/'
    tr_gt_path = '../../UIE_new/DATASET/UIEB/Train/reference/'
    val_input_path='../../UIE_new/DATASET/UIEB/Val/input/'
    val_gt_path = '../../UIE_new/DATASET/UIEB/Val/reference/'
    ts_input_path='../../UIE_new/DATASET/UIEB/Test/input/'
    ts_gt_path = '../../UIE_new/DATASET/UIEB/Test/reference/'
if 'EUVP_Scene' in config['name']:
    tr_input_path='../../UIE_new/DATASET/EUVP_Scenes/Train/input/'
    tr_gt_path = '../../UIE_new/DATASET/EUVP_Scenes/Train/reference/'
    val_input_path='../../UIE_new/DATASET/EUVP_Scenes/Test/input/'
    val_gt_path = '../../UIE_new/DATASET/EUVP_Scenes/Test/reference/'
print(f'input path:{tr_input_path} --- gt path:{tr_gt_path}')
train_lst = get_file_lst(tr_input_path,tr_gt_path)
train_set = UIE_train(train_lst,image_size=config['img_size'])
train_bs = config['batch_size']
train_loader = DataLoader(train_set,batch_size = train_bs,shuffle=True,num_workers=8,pin_memory=True)

print(f'val input:{val_input_path}--- val gt:{val_gt_path}')
val_lst = get_file_lst(val_input_path,val_gt_path)
if args.resize_val:
    val_set = UIE_val(val_lst,image_size=config['img_size'])
else:
    val_set = UIE_val(val_lst,image_size=None)

val_loader = DataLoader(val_set,batch_size = 1,shuffle=False,num_workers=8,pin_memory=True)

in_channels = 3
num_class = 3
num_epochs = config['num_epochs']
lr = config['lr']

net = get_obj_from_string(config['model_file'])(in_channels,num_class,img_size=config['img_size']).to(device)
optimizer = torch.optim.AdamW(net.parameters(),lr = lr,weight_decay=weight_decay)
criterion = nn.MSELoss()

mse_loss_lst = []
cc_mse_loss_lst = []

val_mse_loss_lst = []


max_iters = num_epochs * len(train_loader)


# In[25]:
cc_weight = config['cc_weight']
writer = SummaryWriter(f'{log_path[:-1]}_tensorboard')
def train(net,num_epochs):
    best_val_loss = 1000
    best_val_loss2 = 1000

    for epoch in range(num_epochs):
        net.train()
        tic = time.time()
       
        tr_loss_meter = AverageMeter()
        cc_loss_meter = AverageMeter()
        
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            image = data['image'].to(device)
            label = data['ref'].to(device)
            cc,pred = net(image)
            
            loss_mse = criterion(pred,label)
            cc_loss_mse = criterion(cc,label)
            loss = loss_mse + cc_weight*cc_loss_mse

            tr_loss_meter.update(loss_mse.item(),len(label))
            cc_loss_meter.update(cc_loss_mse.item(),len(label))
            

            loss.backward()
            optimizer.step()
            
        toc = time.time()
        if (epoch+1) % 10 == 0:
            writer.add_scalar('Loss/train',tr_loss_meter.avg,epoch)
            writer.add_scalar('Loss/cc_train',cc_loss_meter.avg,epoch)
        print(f"epoch:{epoch+1}/{num_epochs}, mse loss{tr_loss_meter.avg:.5f}, cc mse loss{cc_loss_meter.avg:.5f}, time:{toc-tic}")

        mse_loss_lst.append(tr_loss_meter.avg)
        cc_mse_loss_lst.append(cc_loss_meter.avg)
        
        np.save(log_path+"tr_mse_loss.npy",np.array(mse_loss_lst))
        np.save(log_path+"tr_cc_mse_loss.npy",np.array(cc_mse_loss_lst))

        net.eval()
        val_loss_meter = AverageMeter()
        tic = time.time()
        with torch.no_grad():
            for data in tqdm(val_loader):
                image = data['image'].to(device)
                label = data['ref'].to(device)
                cc,pred = net(image)                
                loss_mse = criterion(pred,label)
                val_loss_meter.update(loss_mse.item())

        toc = time.time()
        print(f'on val set loss:{val_loss_meter.avg:.5f} time:{toc-tic:.5f}')
        if (epoch+1) % 10 == 0:
            writer.add_scalar('Loss/val',val_loss_meter.avg,epoch)

        val_mse_loss_lst.append(val_loss_meter.avg)
        np.save(log_path+"val_mse_loss.npy",np.array(val_mse_loss_lst))
        if (epoch +1) <= 300:
            if(val_loss_meter.avg < best_val_loss):
                best_val_loss = val_loss_meter.avg
                state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
                torch.save(state_dict, snapshot_path + f'best_val.pth')
                print(f'save best val model on epoch{epoch}')
                f.write(f'save best val model on epoch:{epoch},val loss:{best_val_loss}\n')
        else:
            if(val_loss_meter.avg < best_val_loss2):
                best_val_loss2 = val_loss_meter.avg
                state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
                torch.save(state_dict, snapshot_path + f'best_val2.pth')
                print(f'save best val model on epoch{epoch}')
                f.write(f'save best val model on epoch:{epoch},val loss:{best_val_loss}\n')
        
        state_dict = {'net':net.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
        torch.save(state_dict, snapshot_path + f'latest.pth')

        
    f.close()
    writer.close()
            
train(net,num_epochs)
