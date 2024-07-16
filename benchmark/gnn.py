import os
import random
import numpy as np
import pandas as pd

os.chdir(os.pardir) # to the parent dir

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader #Dataset
from torchmetrics import R2Score

import utils
from utils import kfold_split
import dataloader
from dataloader import Downstream_dataset

from benchmark.gnn_utils import *
import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(path = './benchmark/config_graph.json', print_dict = False)
seed = 72
utils.set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv(config["dataset"], skiprows = 1, header = None)
train_dataset, test_dataset = kfold_split(dataset, k = config["k_fold"], seed = seed)
train_dataset = [Graph_Dataset(train_dataset[i].iloc[:, 0], train_dataset[i].iloc[:, 1]) for i in range(config["k_fold"])]
test_dataset = [Graph_Dataset(test_dataset[i].iloc[:, 0], test_dataset[i].iloc[:, 1] ) for i in range(config["k_fold"])]
train_dataloader = [DataLoader(train_dataset[i], batch_size = config["batch_size"], shuffle = config["shuffle"]) for i in range(config["k_fold"])]
test_dataloader = [DataLoader(train_dataset[i], batch_size = config["batch_size"], shuffle = config["shuffle"]) for i in range(config["k_fold"])]

def train(model, train_dataloader, device, optimizer, loss_fn):
    
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        pred = model(batch)
        prop = batch.y.view(pred.shape).to(torch.float32)

        optimizer.zero_grad()
        loss = loss_fn(pred, prop)
        loss.backward()
        optimizer.step()


def eval(model, device, train_dataloader, test_dataloader, loss_fn, val_dataloader = None):
    model.eval()
    r2score = R2Score()
    train_loss = 0
    test_loss  = 0
    val_loss = 0

    with torch.no_grad():
        train_pred, train_true = torch.tensor([]), torch.tensor([])
        test_pred, test_true = torch.tensor([]), torch.tensor([])
        val_pred, val_true = (torch.tensor([]), torch.tensor([])) if val_dataloader else (None, None)
        
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            pred = model(batch)
            prop = batch.y.view(pred.shape).to(torch.float32)
            loss = loss_fn(pred, prop)
            
            train_loss += loss.detach().cpu().item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), pred.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().detach().cpu(), train_true.flatten().detach().cpu()).item()

        for step, batch in enumerate(test_dataloader):
            batch = batch.to(device)
            pred = model(batch)
            prop = batch.y.view(pred.shape).to(torch.float32)
            loss = loss_fn(pred, prop)

            test_loss += loss.detach().cpu().item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), pred.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])
        
        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().detach().cpu(), test_true.flatten().detach().cpu()).item()

        if val_dataloader:
            for step, batch in enumerate(val_dataloader):
                batch = batch.to(device)
                pred = model(batch)
                prop = batch.y.view(pred.shape).to(torch.float32)
                loss = loss_fn(pred, prop)

                val_loss += loss.detach().cpu().item() * len(prop)
                val_pred = torch.cat([val_pred.to(device), pred.to(device)])
                val_true = torch.cat([val_true.to(device), prop.to(device)])

            val_loss = val_loss / len(val_pred.flatten())
            r2_val = r2score(val_pred.flatten().detach().cpu(), val_true.flatten().detach().cpu()).item()
    if val_dataloader:
        return train_loss, test_loss, val_loss, r2_train, r2_test, r2_val
    else:
        return train_loss, test_loss, r2_train, r2_test
    
train_loss_fold, test_loss_fold, train_r2_fold, test_r2_fold = [], [], [], []   
best_r2 = 0.0
patience = config['patience']

for fold_num, dataloader in enumerate(train_dataloader):
    model = GNN_graphpred(num_layer = config["num_gnn_layers"], emb_dim = config["emb_dim"], num_tasks = config["num_tasks"], JK = config["JK"], 
                          drop_ratio = config["drop_ratio"], graph_pooling = config["graph_pooling"], gnn_type = config["gnn_type"])
    
    loss_fn = nn.MSELoss()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

            
    train_loss_best, test_loss_best, best_train_r2, best_test_r2 = float('inf'), float('inf'), float('-inf'), float('-inf')
    patience_count = 0
    early_stopping_start_epoch = config['early_stopping_start_epoch']

    current_test_dataloader = test_dataloader[fold_num]

    for epoch in range(config['n_epochs']):
        train(model, dataloader, device, optimizer, loss_fn)
        train_loss, test_loss, r2_train, r2_test = eval(model, device, dataloader, current_test_dataloader, loss_fn, val_dataloader = None)
        
        if test_loss < test_loss_best and epoch >= early_stopping_start_epoch:
            train_loss_best = train_loss
            test_loss_best = test_loss
            best_train_r2 = r2_train
            best_test_r2 = r2_test
            patience_count = 0
        elif epoch >= early_stopping_start_epoch:
            patience_count += 1
            
        if r2_test > best_r2:
            best_r2 = r2_test
        
        if patience_count >= config['patience']:
            print(f"Early Stopping at Epoch {epoch + 1}")
            break
        
    train_loss_fold.append(np.sqrt(train_loss_best))
    test_loss_fold.append(np.sqrt(test_loss_best))
    train_r2_fold.append(best_train_r2)
    test_r2_fold.append(best_test_r2)

    #print(f"Best Metrics in Fold {fold_num + 1}/{config['k_fold']} - Train RMSE: {np.sqrt(train_loss_best)} - Test RMSE: {np.sqrt(test_loss_best)} - Train R^2: {best_train_r2} - Test R^2: {best_test_r2}")

'''Average metrics'''
train_rmse = np.mean(np.array(train_loss_fold))
test_rmse = np.mean(np.array(test_loss_fold))
train_r2 = np.mean(np.array(train_r2_fold))  
test_r2 = np.mean(np.array(test_r2_fold))
std_test_rmse = np.std(np.array(test_loss_fold))
std_test_r2 = np.std(np.array(test_r2_fold))

print('Train RMSE = ', train_rmse )
print('Test RMSE = ', test_rmse)
print('Train R2 = ', train_r2)
print('Test R2 = ', test_r2)
print('Std of Test RMSE = ', std_test_rmse)
print('Std of Test R2 = ', std_test_r2)
