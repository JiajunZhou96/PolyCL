# here the fingerprints methods MACCS keys and Morgan fingerprints are used
# XG boost and random forest are used as pure supervised models
#pip3 install -U jupyterlab
import time
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm

os.chdir(os.pardir) # to the parent dir

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchmetrics import R2Score
from torch.cuda.amp import autocast, GradScaler # mixed precision training

import utils
from utils import kfold_split
from dataloader import to_psmiles
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

seed = 72
utils.set_seed(seed)
config = utils.get_config(path = './other_evals/fpmlp.json', print_dict = False)

log_dir_path = "./logs/morgan_mlp/"
if not os.path.exists(log_dir_path):
    os.makedirs(log_dir_path)

log_filename = os.path.join(log_dir_path, f"fpmlp_{os.path.splitext(os.path.basename(config['downstream_dataset']))[0]}_fp_{config['fingerprints_size']}.log")
logging.basicConfig(filename=log_filename, level=logging.INFO, filemode = 'w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info(f"Number of training samples: {len(pd.read_csv(config['downstream_dataset']))}")
logger.info(f"Batch_Size: {config['batch_size']}")
logger.info(f"Downstream Learning_Rate: {config['lr']}")
logger.info(f"Total Number of Epochs: {config['n_epochs']}")


def smiles_to_fp(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)

def train(model, train_dataloader, device, optimizer, loss_fn):
    model.train()
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = [item.to(device) for item in batch]
        X_batch, y_batch = batch
        X_batch = X_batch.float()
        optimizer.zero_grad()
        y_pred = model(X_batch)
        y_batch = y_batch.view(y_pred.shape).to(torch.float32)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def eval(model, device, train_dataloader, test_dataloader, loss_fn, val_dataloader = None):
    model.eval()
    r2score = R2Score()
    train_loss = 0
    test_loss = 0
    val_loss = 0
    
    with torch.no_grad():
        train_pred, train_true = torch.tensor([]), torch.tensor([])
        test_pred, test_true = torch.tensor([]), torch.tensor([])
        val_pred, val_true = (torch.tensor([]), torch.tensor([])) if val_dataloader else (None, None)
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = [item.to(device) for item in batch]
            X_batch, y_batch = batch
            X_batch = X_batch.float()
            y_pred = model(X_batch)
            y_batch = y_batch.view(y_pred.shape).to(torch.float32)
            loss = loss_fn(y_pred, y_batch)
            
            train_loss += loss.detach().cpu().item() * len(y_batch)
            train_pred = torch.cat([train_pred.to(device), y_pred.to(device)])
            train_true = torch.cat([train_true.to(device), y_batch.to(device)])
        
        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().detach().cpu(), train_true.flatten().detach().cpu()).item()
        print('trian RMSE = ', np.sqrt(train_loss))
        print('train loss = ', train_loss)
        print('train r^2 = ', r2_train)
        
        logger.info(f"Total Steps in ONE Epoch: {step + 1}")
        logger.info(f"Train RMSE: {np.sqrt(train_loss):.6f}")
        logger.info(f"Train loss: {train_loss:.6f}")
        logger.info(f"Train R2: {r2_train:.6f}")
        
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = [item.to(device) for item in batch]
            X_batch, y_batch = batch
            X_batch = X_batch.float()
            y_pred = model(X_batch)
            y_batch = y_batch.view(y_pred.shape).to(torch.float32)
            loss = loss_fn(y_pred, y_batch)
            
            test_loss += loss.detach().cpu().item() * len(y_batch)
            test_pred = torch.cat([test_pred.to(device), y_pred.to(device)])
            test_true = torch.cat([test_true.to(device), y_batch.to(device)])
            
        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().detach().cpu(), test_true.flatten().detach().cpu()).item()
        print('test RMSE = ', np.sqrt(test_loss))
        print('test loss = ', test_loss)
        print('test r^2 = ', r2_test)
        
        logger.info(f"Test RMSE: {np.sqrt(test_loss):.6f}")
        logger.info(f"Test loss: {test_loss:.6f}")
        logger.info(f"Test R2: {r2_test:.6f}")
        
        if val_dataloader:
            for step, batch in enumerate(tqdm(val_dataloader)):
                batch = [item.to(device) for item in batch]
                X_batch, y_batch = batch
                X_batch = X_batch.float()
                y_pred = model(X_batch)
                y_batch = y_batch.view(y_pred.shape).to(torch.float32)
                loss = loss_fn(y_pred, y_batch)
                
                val_loss += loss.detach().cpu().item() * len(y_batch)
                val_pred = torch.cat([val_pred.to(device), y_pred.to(device)])
                val_true = torch.cat([val_true.to(device), y_batch.to(device)])
                
            val_loss = val_loss / len(val_pred.flatten())
            r2_val = r2score(val_pred.flatten().detach().cpu(), val_true.flatten().detach().cpu()).item()
            print('val RMSE = ', np.sqrt(val_loss))
            print('val loss = ', val_loss)
            print('val r^2 = ', r2_val)
            
            logger.info(f"Val RMSE: {np.sqrt(val_loss):.6f}")
            logger.info(f"Val loss: {val_loss:.6f}")
            logger.info(f"Val R2: {r2_val:.6f}")
            
    if val_dataloader:
        return train_loss, test_loss, val_loss, r2_train, r2_test, r2_val
    else:
        return train_loss, test_loss, r2_train, r2_test

dataset = pd.read_csv(config["downstream_dataset"], skiprows = 1, header = None)
dataset[0] = dataset[0].apply(to_psmiles)
dataset[2] = dataset[0].apply(lambda x: smiles_to_fp(x, nBits = config['fingerprints_size']))
dataset_fp = pd.DataFrame(list(zip(dataset[2], dataset[1])))

train_dataset, test_dataset = kfold_split(dataset_fp, k = config['k_fold'], seed = config['seed'])

X_train = [torch.tensor(list(train_dataset[i][0])) for i in range(config['k_fold'])]
y_train = [torch.tensor(list(train_dataset[i][1])) for i in range(config['k_fold'])]
X_test = [torch.tensor(list(test_dataset[i][0])) for i in range(config['k_fold'])]
y_test = [torch.tensor(list(test_dataset[i][1])) for i in range(config['k_fold'])]

train_data_torch = [TensorDataset(X_train[i], y_train[i]) for i in range(config['k_fold'])]
test_data_torch = [TensorDataset(X_test[i], y_test[i]) for i in range(config['k_fold'])]

train_dataloader = [DataLoader(train_data_torch[i], batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers']) for i in range(config['k_fold'])]
test_dataloader = [DataLoader(test_data_torch[i], batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers']) for i in range(config['k_fold'])]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, drop_ratio = 0.1, hidden_size = 512, activation_func = nn.ReLU(inplace = True)):
        super(MLP, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation_func,
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        regression = self.regressor(x)
        return regression
    
input_size = config["fingerprints_size"]  
hidden_size = input_size  
output_size = 1 


train_loss_fold, test_loss_fold, train_r2_fold, test_r2_fold = [], [], [], []   
best_r2 = 0.0
patience = config['patience']

for fold_num, dataloader in enumerate(train_dataloader):
    print('Fold %s/%s' % (fold_num + 1, 5))
    model = MLP(input_size = input_size, hidden_size = hidden_size)
    model.to(device)

    loss_fn = nn.MSELoss()  
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])  

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
        #if r2_test > best_test_r2 and epoch >= early_stopping_start_epoch:
            print(f"*******************************New Best Model Found in Fold {fold_num + 1}/{5} at Epoch {epoch + 1}***************************************")
            train_loss_best = train_loss
            test_loss_best = test_loss
            best_train_r2 = r2_train
            best_test_r2 = r2_test
            patience_count = 0
        elif epoch >= early_stopping_start_epoch:
            patience_count += 1

        if r2_test > best_r2:
            best_r2 = r2_test
            #model.save_model(path = config['best_model'])

        if patience_count >= patience:
            print(f"Early Stopping at Epoch {epoch + 1}")
            break


    train_loss_fold.append(np.sqrt(train_loss_best))
    test_loss_fold.append(np.sqrt(test_loss_best))
    train_r2_fold.append(best_train_r2)
    test_r2_fold.append(best_test_r2)

    print(f"Best Metrics in Fold {fold_num + 1}/{5} - Train RMSE: {np.sqrt(train_loss_best)} - Test RMSE: {np.sqrt(test_loss_best)} - Train R^2: {best_train_r2} - Test R^2: {best_test_r2}")
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
    # for parameter in model.parameters():
    #     print(parameter)
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

