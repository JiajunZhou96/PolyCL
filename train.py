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

import utils
import dataloader
import polycl

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = False)
seed = config["seed"]
utils.set_seed(seed)

pretrain_data = utils.read_txt(config["pretrain_data_txt"]) # 这个数据集带星号外带括号，是polybert的数据
pretrain_data = pretrain_data[:1000]


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

def train(model, dataloader1, dataloader2, device, optimizer, n_epochs):
    model.train()
    
    total_batches = len(dataloader1) * n_epochs # nuumber of batches in one epoch * number of epochs
    
    save_every = int(config["save_interval"] * total_batches)
    log_every = int(config["log_interval"] * total_batches)
    
    for epoch in range(1, n_epochs):
        epoch_loss = 0.0
        #start_time = time.time()
        
        for i, batch in enumerate(tqdm(zip(dataloader1, dataloader2), desc= "Iteration", initial= 1)):  # initial= 1 will not affect the initial value of i 

            batch1, batch2 = batch
            
            batch1 = {key: value.to(device) for key, value in batch1.items()}
            batch2 = {key: value.to(device) for key, value in batch2.items()}
            
            optimizer.zero_grad()
            
            _, out1 = model(batch1)
            _, out2 = model(batch2)
            
            out1 = nn.functional.normalize(out1, dim =1)
            out2 = nn.functional.normalize(out2, dim =1)
            loss = ntxent_loss(out1, out2)
            
            
            batches_done = (epoch - 1) * len(dataloader1) + (i + 1)
            #print(batches_done)
            
            if batches_done == 1 or batches_done % log_every == 0:
                #elapsed_time = time.time() - start_time
                avg_loss = epoch_loss / (i + 1)
                print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}")
                #print(f"Epoch {epoch}, Iteration {i + 1}/{len(dataloader1)}, Avg Loss: {avg_loss:.4f}, Elapsed Time: {elapsed_time:.2f}s")
            
            loss.backward()
            optimizer.step()
            
            if batches_done % save_every == 0:
                
                model.save_model(path = f"model/model_epoch{epoch}_batch{i+1}",)
                print(f"Model saved at Epoch [{epoch}], Batch [{i+1}]")
            
            
    #model.save_model(path = config["save_path"])
    print("train_finished and model saved")
    
train(model, dataloader1, dataloader2, device, optimizer, config["n_epochs"])
