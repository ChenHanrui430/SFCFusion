# -*- coding: utf-8 -*-
"""
@author: Hanrui Chen
"""
import torch

from densefuse_net import DenseFuseNet
from utils import test

device = 'cuda'

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load(r'.\train_result\model_weight_new.pkl')['weight'])


test_path ='..\\deep\\'
test(test_path, model, mode='add',img_type='gray')