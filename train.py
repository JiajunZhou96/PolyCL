import os
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import utils
from utils import AverageMeter
from utils import align_loss, uniform_loss
import dataloader
import polycl

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = True)
seed = config["seed"]
utils.set_seed(seed)

pretrain_data = utils.read_txt(config["pretrain_data"])
psmile_data = [dataloader.to_psmiles(smiles) for smiles in pretrain_data]
pretrain_data = psmile_data

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
model = polycl.polyCL(encoder= polyBERT, pooler = config["pooler"])

if config["ckpt"]:
    model.load_state_dict(torch.load(config['ckpt_model']))

# parallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Move to GPUs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ntxent_loss = polycl.NTXentLoss(device = device, batch_size = config["batch_size"], temperature = config["temperature"], use_cosine_similarity = config["use_cosine_similarity"])

optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = utils.get_scheduler(config, optimizer)

# mixed precision
scaler = GradScaler()

# initiation of alignment and uniformity meters
align_meter = AverageMeter('align_loss')
unif_meter = AverageMeter('uniform_loss')

if config["ckpt"]:
    ckpt_dict = torch.load(config['ckpt_dict'])
    optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
    scaler.load_state_dict(ckpt_dict['scaler_state_dict'])
    epoch_start = ckpt_dict['epoch'] + 1

else:
    epoch_start = 1
    

def train(model, dataloader1, dataloader2, device, optimizer, n_epochs):
    model.train()
    
    #n_iter = 1
    total_batches = len(dataloader1) * n_epochs

    save_every = int(config["model_save_interval"] * total_batches)
    log_every  =int(config["log_interval"] * total_batches)
    
    for epoch in range(epoch_start, n_epochs + 1):

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

            if batches_done == list(range(1, log_every, int(log_every/50))) or batches_done % log_every == 0:
                
                avg_loss = epoch_loss / (i + 1)

                rep1_norm = nn.functional.normalize(rep1, dim =1)
                rep2_norm = nn.functional.normalize(rep2, dim =1)

                align_loss_val = align_loss(rep1_norm, rep2_norm, alpha = config["align_alpha"])
                unif_loss_val = (uniform_loss(rep1_norm, t=config["uniformity_t"]) + uniform_loss(rep2_norm, t=config["uniformity_t"])) / 2
                
                print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}, Align Loss: {align_loss_val: .4f}, Unif_Loss: {unif_loss_val: .4f}")

            scaler.scale(loss).backward()
            if config["gradient_clipping"]["enabled"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                max_norm=config["gradient_clipping"]["max_grad_norm"])

            #optimizer.step()
            #Gradient step with GradScaler
            scaler.step(optimizer)
            scaler.update()

            if config["scheduler"]["type"] == "LinearLR":
                scheduler.step(batches_done / total_batches)
            
            else:
                scheduler.step()

            if batches_done % save_every == 0:
                save_dict = {'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'epoch': epoch}
                torch.save(model.module.state_dict(), f"./model/epoch{epoch}_batch{i+1}_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}.pth")
                torch.save(save_dict, f"./model/ckpt_epoch{epoch}_batch{i+1}_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}_dict.pth")
                print(f"Model saved at Epoch [{epoch}], Batch [{i+1}]")
                
    torch.save(model.module.state_dict(), f"./model/final_{len(pretrain_data)}_{config['aug_mode_1']}_{config['aug_mode_2']}_{config['model_dropout']}_{config['batch_size']}_{config['lr']}_{config['scheduler']['type']}.pth")
    print('train_finished and model_saved')

train(model, dataloader1, dataloader2, device, optimizer, config["n_epochs"])
