import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler # mixed precision training

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

pretrain_data = utils.read_txt(config["pretrain_data_txt"]) # 这个数据集带星号外带括号，是polybert的数据
#pretrain_data = pretrain_data[:1000]
psmile_data = [dataloader.to_psmiles(smiles) for smiles in pretrain_data]
pretrain_data = psmile_data

# shuffle the list
random.shuffle(pretrain_data)
pretrain_data1 = deepcopy(pretrain_data)
pretrain_data2 = deepcopy(pretrain_data)

# augmented dataset 1
dataset1 = dataloader.Construct_Dataset(smiles = pretrain_data1, mode = config["aug_mode_1"])
# augmented dataset 2
dataset2 = dataloader.Construct_Dataset(smiles = pretrain_data1, mode = config["aug_mode_2"])

dataloader1 = DataLoader(dataset1, batch_size=config["batch_size"], shuffle=False, drop_last = True)
dataloader2 = DataLoader(dataset2, batch_size=config["batch_size"], shuffle=False, drop_last = True)

polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')
polycl.freeze_layers(polyBERT, layers_to_freeze = config["freeze_layers"])
model = polycl.polyCL(encoder= polyBERT, pooler = config["pooler"])

# parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Move the model to GPUs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ntxent_loss = polycl.NTXentLoss(device = device, batch_size = config["batch_size"], temperature = config["temperature"], use_cosine_similarity = config["use_cosine_similarity"])

#optimizer = optim.Adam(model.parameters(), lr = config["lr"], weight_decay=config["weight_decay"])
optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

# YY LinearLR Scheduler
#scheduler = LinearLR(optimizer, start_factor = 0.5, total_iters = 4)
scheduler = utils.get_scheduler(config, optimizer)

#mixed precision
scaler = GradScaler()

'''alignment and uniformity'''
align_meter = AverageMeter('align_loss')
unif_meter = AverageMeter('uniform_loss')

def train(model, dataloader1, dataloader2, device, optimizer, n_epochs):
    model.train()
    
    #n_iter = 1
    total_batches = len(dataloader1) * n_epochs

    save_every = int(config["save_interval"] * total_batches)
    log_every  =int(config["log_interval"] * total_batches)
    
    for epoch in range(1, n_epochs):

        epoch_loss = 0.0
        align_meter.reset()
        unif_meter.reset()
        
        for i, batch in enumerate(tqdm(zip(dataloader1, dataloader2), desc= "Iteration", initial = 1)):

            batch1, batch2 = batch
            
            batch1 = {key: value.to(device) for key, value in batch1.items()}
            batch2 = {key: value.to(device) for key, value in batch2.items()}
            
            optimizer.zero_grad()

            with autocast():
            
                _, out1 = model(batch1)
                _, out2 = model(batch2)
                
                out1 = nn.functional.normalize(out1, dim =1)
                out2 = nn.functional.normalize(out2, dim =1)
                
                #loss = polycl.NTXentloss(out1, out2)
                loss = ntxent_loss(out1, out2)
            
            # if n_iter % 10 == 0:    
            #     print("train loss", loss.detach().cpu().item())
            epoch_loss += loss.detach().cpu().item()
            align_loss_val = align_loss(out1, out2, alpha = config["align_alpha"])
            unif_loss_val = (uniform_loss(out1, t=config["uniformity_t"]) + uniform_loss(out2, t=config["uniformity_t"])) / 2
            al_un_loss = loss = align_losis_val * config["alignment_w"] + unif_loss_val * config["uniformity_w"]
            batches_done = (epoch - 1) * len(dataloader1) + (i + 1)

            if batches_done == 1 or batches_done % log_every == 0:
                avg_loss = epoch_loss / (i + 1)
                print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}, Align Loss: {align_loss_val: .6f}, Unif_Loss: {unif_loss_val: .6f}, Al_un_loss: {al_un_loss: .6f}")

            #n_iter += 1
            #loss.backward()
            # backward pass with gradient scaling
            scaler.scale(loss).backward()
            ########### gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            if config["gradient_clipping"]["enabled"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                               max_norm=config["gradient_clipping"]["max_grad_norm"])

            #optimizer.step()
            #Gradient step with GradScaler
            scaler.step(optimizer)
            scaler.update()

            ########### LinearLR scheduler
            scheduler.step()

            # if n_iter == 2:
            #     print("training_initiated")

            if batches_done % save_every == 0:
                if isinstance(model, nn.DataParallel):        
                    model.module.save_model(path = f"model/model_epoch{epoch}_batch{i+1}")
                else:
                    model.save_model(path = f"model/model_epoch{epoch}_batch{i+1}")
                print(f"Model saved at Epoch [{epoch}], Batch [{i+1}]")
    print('train_finished and model_saved')

      
train(model, dataloader1, dataloader2, device, optimizer, config["n_epochs"])