# -*- coding: utf-8 -*-
"""

@author: chr
"""
from PIL import Image
import os
import string
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat

_tensor = transforms.ToTensor()
_pil_rgb    = transforms.ToPILImage('RGB')
_pil_gray = transforms.ToPILImage()
device = 'cuda'

def getD(img):
    # img = np.float32(img)
    # t=torch.squeeze(img)
    # tt=torch.std(t).item()
    return torch.std(torch.squeeze(img)).item()

def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def load_img(img_path, img_type='gray'):
    img = Image.open(img_path)
    if img_type=='gray':
        img = img.convert('L')
    return _tensor(img).unsqueeze(0)


class Strategy(nn.Module):
    def __init__(self, mode='add', window_width=1):
        super().__init__()
        self.mode = mode
        if self.mode == 'l1':
            self.window_width = window_width
            
    def forward(self, y1, y2):
        if self.mode == 'add':
            return (y1+y2)/2
        
        if self.mode == 'l1':
            ActivityMap1 = y1.abs()
            ActivityMap2 = y2.abs()
            
            kernel = torch.ones(2*self.window_width+1,2*self.window_width+1)/(2*self.window_width+1)**2
            kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
            kernel = kernel.expand(y1.shape[1],y1.shape[1],2*self.window_width+1,2*self.window_width+1)
            ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=self.window_width)
            ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=self.window_width)
            WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
            WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
            return WeightMap1*y1+WeightMap2*y2

def fusion(x1,x2,x3,x4,x5,x6,model,mode='l1', window_width=1,D=None):
    with torch.no_grad():
        fusion_layer  = Strategy(mode,window_width).to(device)
        feature1 = model.encoder(x1) # base vis
        feature2 = model.encoder(x2) # base ir
        feature3 = model.encoder(x3) # saliency vis
        feature4 = model.encoder(x4) # saliency ir
        feature5 = model.encoder(x5) # source vis
        feature6 = model.encoder(x6) # source ir
        if D is not None:
            W = 1+(torch.sqrt(D[2]+D[3]) - torch.sqrt(D[0]+D[1]))/torch.sqrt(D[0]+D[1])
            w1 = W*(0.5+(D[2]-D[3])/(D[0]+D[1]))
            w2 = W*(0.5+(D[3]-D[2])/(D[0]+D[1]))
            #
            featureSF=min(D[2],D[3])/(D[2]+D[3]) * torch.where(feature3 - feature4>=0.0, feature3, feature4)+max(D[2],D[3])/(D[2]+D[3])*torch.where(feature3 - feature4<0.0, feature3, feature4)
            weightMapMax = torch.where(feature1 >= feature2, feature1, feature2)
            weightMapMax_final = torch.where(torch.isnan(weightMapMax), torch.full_like(weightMapMax, 1), weightMapMax)
            weightMapMin = torch.where(feature1 <= feature2, feature1, feature2)
            weightMapMin_final = torch.where(torch.isnan(weightMapMin), torch.full_like(weightMapMin, 0), weightMapMin)
            featureSBF = featureSF+weightMapMin_final*(weightMapMax_final-weightMapMin_final)/(weightMapMax_final.add(1))
            feature_fusion = featureSBF+ (w1/W * feature5 +w2/W * feature6)-(w1*feature3+w2*feature4)
        else:
            feature7=feature5
            feature8=feature6
            feature_fusion = fusion_layer(feature7, feature8)
        return model.decoder(feature_fusion).squeeze(0).detach().cpu()

class Test:
    def __init__(self):
        pass
        
    def load_imgs(self, img1_path,img2_path, device):
        img1 = load_img(img1_path,img_type=self.img_type).to(device)
        img2 = load_img(img2_path,img_type=self.img_type).to(device)
        return img1, img2
    
    def save_imgs(self, save_path,save_name, img_fusion):
        mkdir(save_path)
        save_path = os.path.join(save_path,save_name)
        img_fusion.save(save_path)

class test_gray(Test):
    def __init__(self):
        self.img_type = 'gray'
    
    def get_fusion(self,img1_path,img2_path,img3_path,img4_path,img5_path,img6_path,model,
                   save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
        img1, img2 = self.load_imgs(img1_path,img2_path,device)
        img3, img4 = self.load_imgs(img3_path,img4_path,device)
        img5, img6 = self.load_imgs(img5_path,img6_path,device)
        Dlist =torch.rand(4)
        Dlist=Dlist.cuda()
        for i,v in enumerate([img5,img6,img1,img2]):
            Dlist[i]=getD(v)

        img_fusion = fusion(x1=img1,x2=img2,x3=img3,x4=img4,x5=img5,x6=img6,model=model,mode=mode,window_width=window_width,D=Dlist)
        img_fusion = _pil_gray(img_fusion)
        
        self.save_imgs(save_path,save_name, img_fusion)
        return img_fusion

class test_rgb(Test):
    def __init__(self):
        self.img_type = 'rgb'
        
    def get_fusion(self,img1_path,img2_path,model,
                   save_path = './test_result/', save_name = 'none', mode='l1',window_width=1):
        img1, img2 = self.load_imgs(img1_path,img2_path,device)
        
        img_fusion = _pil_rgb(torch.cat(
                             [fusion(img1[:,i,:,:][:,None,:,:], 
                             img2[:,i,:,:][:,None,:,:], model,
                             mode=mode,window_width=window_width) 
                             for i in range(3)],
                            dim=0))
                             
        self.save_imgs(save_path,save_name, img_fusion)
        return img_fusion
    
    
def test(test_path, model, img_type='gray', save_path='./test_result/',mode='l1',window_width=1):
    img_list = glob(test_path+'*')
    img_num = len(img_list)/6
    suffix = img_list[0].split('.')[-1]
    img_name_list = sorted(list(set([img_list[i].split('/')[-1].split('.')[0].strip(string.digits) for i in range(len(img_list))])))
    
    if img_type == 'gray':    
        fusion_phase = test_gray()
    elif img_type == 'rgb':
        fusion_phase = test_rgb()
    
    for i in range(int(img_num)):
        img1_path = test_path+img_name_list[1]+str(i+1)+'.'+suffix
        img2_path = test_path+img_name_list[0]+str(i+1)+'.'+suffix
        img3_path = test_path+img_name_list[3]+str(i+1)+'.'+suffix
        img4_path = test_path+img_name_list[2]+str(i+1)+'.'+suffix
        img5_path = test_path+img_name_list[5]+str(i+1)+'.'+suffix #source vis
        img6_path = test_path+img_name_list[4]+str(i+1)+'.'+suffix # source ir
        save_name = str(i+1)+'.'+'png'
        fusion_phase.get_fusion(img1_path,img2_path,img3_path,img4_path,img5_path,img6_path,model,
                   save_path = save_path, save_name = save_name, mode=mode,window_width=window_width)
