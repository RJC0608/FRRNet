import cv2
import torch
import torch.nn.functional as F
import sys
import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.FRRNet import Net

from data import get_loader
from utils import clip_gradient, adjust_lr
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

#-------------------dataset root--------------------------------
# image_root1 = 'F:\\3\\test\\image\\'
# gt_root1 = 'F:\\3\\test\\groundtruth\\'

image_root1 = 'F:\\Dataset\\CDS2K\\test\\Image\\'
gt_root1 = 'F:\\Dataset\\CDS2K\\test\\GT\\'
#-------------------model & path--------------------------------
model = Net()
model.load_state_dict(torch.load('F:\\BBNet--CoCOD-main\\pthKCNet\\CDS2K\\train_w.pth.59'))
model.cuda()
model.eval()

test_datasets=os.listdir(image_root1)

for dataset in test_datasets:
    print(dataset)

    image_root = image_root1+dataset+'/'
    gt_root = gt_root1+dataset+'/'
    test_loader = test_dataset(image_root, gt_root,opt.testsize)  

    save_path = './pred3/CDS2K/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()        
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        #res, S_3_pred, S_4_pred, S_5_pred, S_g_pred = model(image)
        res, res2, res3, res4 = model(image)
        #s4321, edge_mapu,  out4321 = model(image)
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
        cv2.imwrite(save_path+name, res*255)

    
    




