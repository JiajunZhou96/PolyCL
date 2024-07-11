import os
import logging
import random
import numpy as np
import pandas as pd

os.chdir(os.pardir)

from copy import deepcopy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler

import utils
from utils import AverageMeter
from utils import align_loss, uniform_loss
import dataloader
import polycl

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = False)
seed = config["seed"]
utils.set_seed(seed)

test_data = utils.read_txt("./datasets/60k_random_dev.txt")
test_data1 = deepcopy(test_data)
test_data2 = deepcopy(test_data)
dataset1 = dataloader.Construct_Dataset(smiles = test_data1, mode = "original")
dataset2 = dataloader.Construct_Dataset(smiles = test_data2, mode = "enumeration")
dataloader1 = DataLoader(dataset1, batch_size= 64, shuffle=False)
dataloader2 = DataLoader(dataset2, batch_size= 64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = polycl.set_dropout(AutoConfig.from_pretrained('kuelumbus/polyBERT'), dropout = False)
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT', config = model_config)
polyBERT.eval()
for param in polyBERT.parameters():
    param.requires_grad = False
PretrainedModel = polycl.polyCL(encoder = polyBERT, pooler = config['pooler'])
PretrainedModel.to(device)
PretrainedModel.eval()

align_meter = AverageMeter('align_loss')
unif_meter = AverageMeter('uniform_loss')

rep1_norm_s = []
rep2_norm_s = []

for i, batch in enumerate(zip(dataloader1, dataloader2)):
    
    batch1, batch2 = batch
    batch1 = {key: value.to(device) for key, value in batch1.items()}
    batch2 = {key: value.to(device) for key, value in batch2.items()}
    
    rep1, _ = PretrainedModel(batch1)
    rep2, _ = PretrainedModel(batch2)
    rep1_norm = nn.functional.normalize(rep1, dim =1)
    rep2_norm = nn.functional.normalize(rep2, dim =1)
    
    rep1_norm_s.append(rep1_norm.cpu().detach())
    rep2_norm_s.append(rep2_norm.cpu().detach())

rep_1 = torch.cat(rep1_norm_s, dim=0)
rep_2 = torch.cat(rep2_norm_s, dim=0)

align_loss_val = align_loss(rep_1, rep_2, alpha = 2)
unif_loss_val = (uniform_loss(rep_1, t=2) + uniform_loss(rep_2, t=2)) / 2
print(align_loss_val, "align_loss_val")
print(unif_loss_val, "unif_loss_val")
