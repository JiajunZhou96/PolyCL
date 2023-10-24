import os
import logging
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig
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

pretrain_data = utils.read_txt(config["pretrain_data"])
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


model_config = polycl.set_dropout(AutoConfig.from_pretrained('kuelumbus/polyBERT'), dropout = config["model_dropout"])
polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT', config = model_config)
polycl.freeze_layers(polyBERT, layers_to_freeze = config["freeze_layers"], freeze_layer_dropout = False)
model = polycl.polyCL(encoder= polyBERT, pooler = config["pooler"])

# Move the model to GPUs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

ntxent_loss = polycl.NTXentLoss(device = device, batch_size = config["batch_size"], temperature = config["temperature"], use_cosine_similarity = config["use_cosine_similarity"])

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = utils.get_scheduler(config, optimizer)

# mixed precision
scaler = GradScaler()

# create a logger
logging.basicConfig(filename=f"/home/mmm1303/PolyCL/log/{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_{config['n_epochs']}_{config['temperature']}.log", 
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# initiation of alignment and uniformity meters
align_meter = AverageMeter('align_loss')
unif_meter = AverageMeter('uniform_loss')

logger.info(f"Number of training samples: {len(pretrain_data)}")
logger.info(f"Augmentation Mode 1: {config['aug_mode_1']}")
logger.info(f"Augmentation Mode 2: {config['aug_mode_2']}")
logger.info(f"Model_Dropout: {config['model_dropout']}")
logger.info(f"Batch_Size: {config['batch_size']}")
logger.info(f"Maximum Learning_Rate: {config['lr']}")
logger.info(f"Learning Rate Scheduler: {config['scheduler']['type']}")
logger.info(f"Total Number of Epochs: {config['n_epochs']}")
logger.info(f"NTXENT Temperature: {config['temperature']}")

def train(model, dataloader1, dataloader2, device, optimizer, n_epochs):
    model.train()
    
    #n_iter = 1
    total_batches = len(dataloader1) * n_epochs

    save_every = int(config["model_save_interval"] * total_batches)
    log_every  =int(config["log_interval"] * total_batches)
    
    for epoch in range(1, n_epochs + 1):

        epoch_loss = 0.0
        align_meter.reset()
        unif_meter.reset()
        
        for i, batch in enumerate(tqdm(zip(dataloader1, dataloader2), desc= "Iteration", initial = 1)):

            batch1, batch2 = batch
            
            batch1 = {key: value.to(device) for key, value in batch1.items()}
            batch2 = {key: value.to(device) for key, value in batch2.items()}
            
            optimizer.zero_grad()

            with autocast():
            
                rep1, out1 = model(batch1)
                rep2, out2 = model(batch2)
                
                out1 = nn.functional.normalize(out1, dim =1)
                out2 = nn.functional.normalize(out2, dim =1)
                
                #loss = polycl.NTXentloss(out1, out2)
                loss = ntxent_loss(out1, out2)
            
            epoch_loss += loss.detach().cpu().item()
            batches_done = (epoch - 1) * len(dataloader1) + (i + 1)

            if batches_done == list(range(1, log_every, int(log_every/50))) or batches_done % log_every == 0:  # 一开始记录频繁一点 # check int(log_every/50) 对不对
                
                avg_loss = epoch_loss / (i + 1)

                rep1_norm = nn.functional.normalize(rep1, dim =1)
                rep2_norm = nn.functional.normalize(rep2, dim =1)

                align_loss_val = align_loss(rep1_norm, rep2_norm, alpha = config["align_alpha"])
                unif_loss_val = (uniform_loss(rep1_norm, t=config["uniformity_t"]) + uniform_loss(rep2_norm, t=config["uniformity_t"])) / 2
                #al_un_loss = loss = align_loss_val * config["alignment_w"] + unif_loss_val * config["uniformity_w"]
                
                print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}, Align Loss: {align_loss_val: .4f}, Unif_Loss: {unif_loss_val: .4f}")
                
                logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}")
                logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Align Loss: {align_loss_val:.4f}, Unif_Loss: {unif_loss_val: .4f}")
                logger.info(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, learning_rate_current_epoch: {optimizer.param_groups[0]['lr']}")
                #print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}, Align Loss: {align_loss_val: .4f}, Unif_Loss: {unif_loss_val: .4f}, Al_un_loss: {al_un_loss: .4f}")

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
            if config["scheduler"]["type"] == "LinearLR":
                scheduler.step(batches_done / total_batches)
            
            else:
                scheduler.step()

            if batches_done % save_every == 0:
                model.save_model(path = f"/home/mmm1303/PolyCL/model/epoch{epoch-1}_batch{i+1}_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_{config['n_epochs']}_{config['temperature']}.pth")
                
                logger.info(f"Intermediate model saved at Epoch [{epoch-1}], Batch [{i+1}] with name: model/epoch{epoch-1}_batch{i+1}_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_{config['n_epochs']}_{config['temperature']}.pth")
                print(f"Model saved at Epoch [{epoch-1}], Batch [{i+1}]")
                
    model.save_model(path = f"/home/mmm1303/PolyCL/model/final_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_{config['n_epochs']}_{config['temperature']}.pth")
    
    logger.info(f"Final model saved with name: model/final_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_{config['n_epochs']}_{config['temperature']}.pth")
    print('train_finished and model_saved')


train(model, dataloader1, dataloader2, device, optimizer, config["n_epochs"])
