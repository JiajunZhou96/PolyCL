'''https://github.com/Shen-Lab/GraphCL/blob/master/transferLearning_MoleculeNet_PPI/chem/finetune.py'''

'''https://github.com/yuyangw/MolCLR/blob/master/finetune.py'''

'''https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py'''

import os
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import logging

os.chdir(os.pardir) # to the parent dir

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import R2Score

import utils
from utils import kfold_split
import dataloader
from dataloader import Downstream_dataset, to_psmiles
import polycl

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(path = './config_tf_notebook.json', print_dict = False)
seed = config["seed"]
utils.set_seed(seed)

#activation function
activation_functions = {
    'relu': nn.ReLU(inplace=True),
    'prelu': nn.PReLU(),
    'leakyrelu': nn.LeakyReLU(),
    'selu': nn.SELU(),
    'silu': nn.SiLU(),
    'gelu': nn.GELU()
}
activation = activation_functions.get(config['activation'])

# Move the model to GPUs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a logger
log_dir_path = './logs/polybert/'
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)
log_filename = os.path.join(log_dir_path, f"polybert_{os.path.splitext(os.path.basename(config['downstream_dataset']))[0]}_{config['activation']}_lr_{config['lr']}_{config['batch_size']}_nepochs_{config['n_epochs']}_dropout_{config['dropout_ratio']}_{config['k_fold']}_hidden_{config['hidden_size']}_decay_{config['weight_decay']}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, filemode = 'w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info(f"Number of training samples: {len(pd.read_csv(config['downstream_dataset']))}")
logger.info(f"Batch_Size: {config['batch_size']}")
logger.info(f"Downstream Learning_Rate: {config['lr']}")
logger.info(f"Total Number of Epochs: {config['n_epochs']}")


def train(model, train_dataloader, device, optimizer, loss_fn):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc = 'Iteration')):
        batch = {key: value.to(device) for key, value in batch.items()}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pred = model(input_ids = input_ids, attention_mask = attention_mask, data = None)
        prop = batch['labels'].view(pred.shape).to(torch.float32)

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
        
        for step, batch in enumerate(tqdm(train_dataloader, desc = 'Training Iteration')):
            batch = {key: value.to(device) for key, value in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pred = model(input_ids = input_ids, attention_mask = attention_mask, data=None)
            prop = batch['labels'].view(pred.shape).to(torch.float32)
            loss = loss_fn(pred, prop)
            
            train_loss += loss.detach().cpu().item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), pred.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().detach().cpu(), train_true.flatten().detach().cpu()).item()
        print('trian RMSE = ', np.sqrt(train_loss))
        print('train loss = ', train_loss)
        print('train r^2 = ', r2_train)
        
        logger.info(f"Step: {step + 1}")
        logger.info(f"Train RMSE: {np.sqrt(train_loss):.6f}")
        logger.info(f"Train loss: {train_loss:.6f}")
        logger.info(f"Train R2: {r2_train:.6f}")

        for step, batch in enumerate(tqdm(test_dataloader, desc = 'Test Iteration')):
            batch = {key: value.to(device) for key, value in batch.items()} 
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pred = model(input_ids = input_ids, attention_mask = attention_mask, data = None)
            prop = batch['labels'].view(pred.shape).to(torch.float32)
            loss = loss_fn(pred, prop)

            test_loss += loss.detach().cpu().item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), pred.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])
        
        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().detach().cpu(), test_true.flatten().detach().cpu()).item()
        print('test RMSE = ', np.sqrt(test_loss))
        print('test loss = ', test_loss)
        print('test r^2 = ', r2_test)
        
        logger.info(f"Test RMSE: {np.sqrt(test_loss):.6f}")
        logger.info(f"Test loss: {test_loss:.6f}")
        logger.info(f"Test R2: {r2_test:.6f}")

        if val_dataloader:
            for batch in enumerate(tqdm(val_dataloader, desc = 'Valid Iteration')):
                batch = {key: value.to(device) for key, value in batch.items()}
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                pred = model(input_ids = input_ids, attention_mask = attention_mask, data = None)
                prop = batch['labels'].view(pred.shape).to(torch.float32)
                loss = loss_fn(pred, prop)

                val_loss += loss.detach().cpu().item() * len(prop)
                val_pred = torch.cat([val_pred.to(device), pred.to(device)])
                val_true = torch.cat([val_true.to(device), prop.to(device)])

            val_loss = val_loss / len(val_pred.flatten())
            r2_val = r2score(val_pred.flatten().detach().cpu(), val_true.flatten().detach().cpu()).item()
            print('Val RMSE = ', np.sqrt(val_loss))
            print('Val loss = ', val_loss)
            print('Val r^2 = ', r2_val)

    if val_dataloader:
        return train_loss, test_loss, val_loss, r2_train, r2_test, r2_val
    else:
        return train_loss, test_loss, r2_train, r2_test

def main(config):

    print('K Fold spliting')
    dataset = pd.read_csv(config['downstream_dataset'], skiprows = 1, header = None)
    dataset[0] = dataset[0].apply(to_psmiles)
    train_dataset, test_dataset = kfold_split(dataset, k = config['k_fold'], seed = config['seed'])

    tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
    
    train_dataset_down = [Downstream_dataset(train_dataset[i], block_size = config['block_size'], tokenizer = tokenizer) for i in range(config['k_fold'])]
    test_dataset_down = [Downstream_dataset(test_dataset[i], block_size = config['block_size'], tokenizer = tokenizer) for i in range(config['k_fold'])]
    train_dataloader = [DataLoader(train_dataset_down[i], batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers']) for i in range(config['k_fold'])]
    test_dataloader = [DataLoader(test_dataset_down[i], batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers']) for i in range(config['k_fold'])]     
    
    train_loss_fold, test_loss_fold, train_r2_fold, test_r2_fold = [], [], [], []   
    best_r2 = 0.0
    patience = config['patience']
    for fold_num, dataloader in enumerate(train_dataloader):
        print('Fold %s/%s' % (fold_num + 1, config['k_fold']))
        logger.info(f"Starting Fold {fold_num + 1}/{config['k_fold']}")
        
        model_config = polycl.set_dropout(AutoConfig.from_pretrained('kuelumbus/polyBERT'), dropout = False)
        polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT', config = model_config)
        for param in polyBERT.parameters():
            param.requires_grad = False
        PretrainedModel = polyBERT
        PretrainedModel.eval()
        model  = polycl.polycl_pred(PretrainedModel = PretrainedModel, drop_ratio = config['dropout_ratio'], activation_func = activation)
        loss_fn = nn.MSELoss()
        model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        torch.cuda.empty_cache()

        train_loss_best, test_loss_best, best_train_r2, best_test_r2 = float('inf'), float('inf'), float('-inf'), float('-inf')
        patience_count = 0
        early_stopping_start_epoch = config['early_stopping_start_epoch']

        current_test_dataloader = test_dataloader[fold_num]
        
        for epoch in range(config['n_epochs']):
            print("epoch: %s/%s" % (epoch + 1, config['n_epochs']))
            logger.info(f"Epoch {epoch + 1}/{config['n_epochs']} in Fold {fold_num + 1}/{config['k_fold']}")
            train(model, dataloader, device, optimizer, loss_fn)
            train_loss, test_loss, r2_train, r2_test = eval(model, device, dataloader, current_test_dataloader, loss_fn, val_dataloader = None)

            if test_loss < test_loss_best and epoch >= early_stopping_start_epoch:
                print(f"*******************************New Best Model Found in Fold {fold_num + 1}/{config['k_fold']} at Epoch {epoch + 1}***************************************")
                train_loss_best = train_loss
                test_loss_best = test_loss
                best_train_r2 = r2_train
                best_test_r2 = r2_test
                patience_count = 0
            elif epoch >= early_stopping_start_epoch:
                patience_count += 1

            if r2_test > best_r2:
                best_r2 = r2_test
                #torch.save(model.module.state_dict(), path = config['best_model'])

            if patience_count >= config['patience']:
                print(f"Early Stopping at Epoch {epoch + 1}")
                break

        train_loss_fold.append(np.sqrt(train_loss_best))
        test_loss_fold.append(np.sqrt(test_loss_best))
        train_r2_fold.append(best_train_r2)
        test_r2_fold.append(best_test_r2)

        print(f"Best Metrics in Fold {fold_num + 1}/{config['k_fold']} - Train RMSE: {np.sqrt(train_loss_best)} - Test RMSE: {np.sqrt(test_loss_best)} - Train R^2: {best_train_r2} - Test R^2: {best_test_r2}")
        logger.info(f"Best Metrics in Fold {fold_num + 1}/{config['k_fold']} - Train RMSE: {np.sqrt(train_loss_best)} - Test RMSE: {np.sqrt(test_loss_best)} - Train R^2: {best_train_r2} - Test R^2: {best_test_r2}")

    train_rmse = np.mean(np.array(train_loss_fold))
    test_rmse = np.mean(np.array(test_loss_fold))
    train_r2 = np.mean(np.array(train_r2_fold))  
    test_r2 = np.mean(np.array(test_r2_fold))
    std_test_rmse = np.std(np.array(test_loss_fold))
    std_test_r2 = np.std(np.array(test_r2_fold))
    
    logger.info(f"Average Metrics after {config['k_fold']} Folds - Train RMSE: {train_rmse} - Test RMSE: {test_rmse} - Train R^2: {train_r2} - Test R^2: {test_r2}")
    logger.info(f"Standard Deviation of Test RMSE: {std_test_rmse}")
    logger.info(f"Standard Deviation of Test R^2: {std_test_r2}")

    print('Train RMSE = ', train_rmse )
    print('Test RMSE = ', test_rmse)
    print('Train R2 = ', train_r2)
    print('Test R2 = ', test_r2)
    print('Std of Test RMSE = ', std_test_rmse)
    print('Std of Test R2 = ', std_test_r2)

if __name__ == "__main__":
    main(config)
    
