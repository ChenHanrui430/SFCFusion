# -*- coding: utf-8 -*-
"""


@author:chr
"""
import torch

from net import DenseFuseNet
from utils import test

device = 'cuda'

model = DenseFuseNet().to(device)
model.load_state_dict(torch.load('./train_result/model_weight.pkl')['weight'])

test_path ='./images/TNODeepPNGNSST/' 
test(test_path, model, mode='add',img_type='gray')
