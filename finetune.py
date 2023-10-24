'''https://github.com/Shen-Lab/GraphCL/blob/master/transferLearning_MoleculeNet_PPI/chem/finetune.py'''

'''https://github.com/yuyangw/MolCLR/blob/master/finetune.py'''

'''https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py'''

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
from torchmetrics import R2Score
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast, GradScaler # mixed precision training

import utils
from utils import AverageMeter
from utils import align_loss, uniform_loss
from utils import scaffold_split, random_split, kfold_split
import dataloader
from dataloader import Downstream_dataset, Normalizer
import polycl

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import warnings
warnings.filterwarnings("ignore")

config = utils.get_config(print_dict = False)
seed = config["seed"]
utils.set_seed(seed)

# Move the model to GPUs 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#criterion = nn.BCEWithLogitsloss(reduction = None)

def train(model, train_dataloader, device, optimizer, loss_fn, normalizer):
    model.train()
    #normalizer = None

    for step, batch in enumerate(tqdm(train_dataloader, desc = 'Iteration')):
        #batch = batch.to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        pred = model(batch) #model prediction
        prop = batch['labels'].view(pred.shape).to(torch.float32) #ground truth

        optimizer.zero_grad()
        if config['label_normalized']:
            loss = loss_fn(pred, normalizer.norm(prop))
        else:
            loss = loss_fn(pred, prop)
        loss.backward()
        optimizer.step()
    #return None

def eval(model, device, train_dataloader, test_dataloader, loss_fn, val_dataloader = None, normalizer = None):
    model.eval()
    # y_true = []
    # y_scores = []
    r2score = R2Score()
    train_loss = 0
    test_loss  = 0
    val_loss = 0

#'''regression'''

    with torch.no_grad():
        train_pred, train_true = torch.tensor([]), torch.tensor([])
        test_pred, test_true = torch.tensor([]), torch.tensor([])
        val_pred, val_true = (torch.tensor([]), torch.tensor([])) if val_dataloader else (None, None)
        
        for step, batch in enumerate(tqdm(train_dataloader, desc = 'Training Iteration')):
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(batch)
            prop = batch['labels'].to(device).float()
            if config['label_normalized']:
                loss = loss_fn(pred, normalizer.denorm(prop))
            else:
                loss = loss_fn(pred, prop)
            
            train_loss += loss.detach().cpu().item() * len(prop)
            train_pred = torch.cat([train_pred.to(device), pred.to(device)])
            train_true = torch.cat([train_true.to(device), prop.to(device)])

        train_loss = train_loss / len(train_pred.flatten())
        r2_train = r2score(train_pred.flatten().detach().cpu(), train_true.flatten().detach().cpu()).item()
        print('trian RMSE = ', np.sqrt(train_loss))
        print('train r^2 = ', r2_train)

        for step, batch in enumerate(tqdm(test_dataloader, desc = 'Test Iteration')):
            batch = {key: value.to(device) for key, value in batch.items()} 
            pred = model(batch)
            prop = batch['labels'].to(device).float()
            if config['label_normalized']:
                loss = loss_fn(pred, normalizer.denorm(prop))
            else:
                loss = loss_fn(pred, prop)

            test_loss += loss.detach().cpu().item() * len(prop)
            test_pred = torch.cat([test_pred.to(device), pred.to(device)])
            test_true = torch.cat([test_true.to(device), prop.to(device)])
        
        test_loss = test_loss / len(test_pred.flatten())
        r2_test = r2score(test_pred.flatten().detach().cpu(), test_true.flatten().detach().cpu()).item()
        print('test RMSE = ', np.sqrt(test_loss))
        print('test r^2 = ', r2_test)

        if val_dataloader:
            for batch in enumerate(tqdm(val_dataloader, desc = 'Valid Iteration')):
                batch = {key: value.to(device) for key, value in batch.items()} 
                pred = model(batch)
                prop = batch['labels'].to(device).float()
                if config['label_normalized']:
                    loss = loss_fn(pred, Normalizer.denorm(prop))
                else:
                    loss = loss_fn(pred, prop)

                val_loss += loss.detach().cpu().item() * len(prop)
                val_pred = torch.cat([val_pred.to(device), pred.to(device)])
                val_true = torch.cat([val_true.to(device), prop.to(device)])

            val_loss = val_loss / len(val_pred.flatten())
            r2_val = r2score(val_pred.flatten().detach().cpu(), val_true.flatten().detach().cpu()).item()
            print('test RMSE = ', np.sqrt(val_loss))
            print('test r^2 = ', r2_val)
    
    model.module.save_model(path = config['eval_model_ckpt'])

    if val_dataloader:
        return train_loss, test_loss, val_loss, r2_train, r2_test, r2_val
    else:
        return train_loss, test_loss, r2_train, r2_test

'''multilabel classification'''
'''
    for step, batch in enumerate(tqdm(train_dataloader, desc = 'Iteration')):
        batch = batch.to(device)

        with torch.no_grad():
            _, pred = model(batch)

        y_true.append(batch['labels'].view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1 ) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print('Some target is missing')
'''


def main(config):

    if config['split'] == 'scaffold' or config['split'] == 'random':
        dataset = pd.read_csv(config['downstream_dataset'], skiprows = 1, header = None)
        mol_list = pd.read_csv(config['downstream_dataset'], header = None)[0].tolist()
        if config['split'] == 'scaffold':
            train_dataset, val_dataset, test_dataset = scaffold_split(dataset, mol_list, train_frac = 0.8, val_frac = 0.1, test_frac = 0.1)
            print('Scaffold splitting')
        elif config['split'] == 'random':
            train_dataset, val_dataset, test_dataset = random_split(dataset, mol_list, train_frac = 0.8, val_frac = 0.1, test_frac = 0.1)
            print('Random splitting')
        else:
            print('Invalid split option')

        train_dataset_down = Downstream_dataset(train_dataset, block_size = config['block_size'])
        test_dataset_down = Downstream_dataset(test_dataset, block_size = config['block_size'])
        val_dataset_down = Downstream_dataset(val_dataset, block_size = config['block_size'])
        train_dataloader = DataLoader(train_dataset_down, batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers'])
        test_dataloader = DataLoader(test_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])
        val_dataloader = DataLoader(val_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])

        polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')
        polycl.freeze_layers(polyBERT, layers_to_freeze = config["freeze_layers"])
        model = polycl.polycl_pred(encoder = polyBERT, pooler = config['pooler'], drop_ratio = config['dropout_ratio'])
        model.from_pretrained(config['pretrained_model_path'])
        loss_fn = nn.MSELoss()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        train_labels = []
        for batch in train_dataloader:
            train_labels.append(batch['labels'])
        train_labels_tensor = torch.cat(train_labels, dim=0)
        # Instantiate the Normalizer
        normalizer = Normalizer(train_labels_tensor)
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

        torch.cuda.empty_cache()
        best_r2 = 0.0
        train_loss_best, test_loss_best, val_loss_best, best_train_r2, best_test_r2, best_val_r2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        train_loss_avg, test_loss_avg, val_loss_avg, train_r2_avg, test_r2_avg, val_r2_avg = [], [], [], [], [], []
        count = 0
        for epoch in range(config['n_epochs']):
            print("epoch: %s/$s" % (epoch + 1, config['n_epochs']))
            train(model, train_dataloader, device, optimizer, loss_fn, normalizer)
            train_loss, test_loss, val_loss, r2_train, r2_test, r2_val = eval(model, device, train_dataloader, test_dataloader, loss_fn, val_dataloader = val_dataloader, normalizer = normalizer)

            if r2_val > best_val_r2:
                best_train_r2 = r2_train
                best_test_r2 = r2_test
                best_val_r2 = r2_val
                train_loss_best = train_loss
                test_loss_best = test_loss
                val_loss_best = val_loss
                count = 0
            else:
                count += 1

            if r2_val > best_r2:
                best_r2 = r2_val
                model.save_model(path = config['best_model'])

            if count >= config['tolerance']:
                print('Early stop')
                if best_val_r2 == 0:
                    print('Bad performance with r^2 < 0')
                break

        train_loss_avg.append(np.sqrt(train_loss_best))
        test_loss_avg.append(np.sqrt(test_loss_best))
        val_loss_avg.append(np.sqrt(val_loss_best))
        train_r2_avg.append(best_train_r2)
        test_r2_avg.append(best_test_r2)
        val_r2_avg.append(best_val_r2)

        '''Average metrics'''
        train_rmse = np.mean(np.array(train_loss_avg))
        test_rmse = np.mean(np.array(test_loss_avg))
        val_rmse = np.mean(np.array(val_loss_avg))
        train_r2 = np.mean(np.array(train_r2_avg))
        test_r2 = np.mean(np.array(test_r2_avg))
        val_r2 = np.mean(np.array(val_r2_avg))
        std_val_rmse = np.std(np.array(val_loss_avg))
        std_val_r2 = np.std(np.array(val_r2_avg))

        print('Train RMSE = ', train_rmse )
        print('Test RMSE = ', test_rmse)
        print('Val RMSE = ', val_rmse)
        print('Train R2 = ', train_r2)
        print('Test R2 = ', test_r2)
        print('Val R2 = ', val_r2)
        print('Std of Valid RMSE = ', std_val_rmse)
        print('Std of Valid R2 = ', std_val_r2)


    # elif config['split'] == 'random':
    #     dataset = pd.read_csv(config['downstream_dataset'], skiprows = 1, header = None)
    #     mol_list = pd.read_csv(config['downstream_dataset'], header = None)[0].tolist()
    #     train_dataset, val_dataset, test_dataset = random_split(dataset, mol_list, train_frac = 0.8, val_frac = 0.1, test_frac = 0.1)
    #     print('Random splitting')

    #     train_dataset_down = Downstream_dataset(train_dataset, block_size = config['block_size'])   
    #     test_dataset_down = Downstream_dataset(test_dataset, block_size = config['block_size']) 
    #     val_dataset_down = Downstream_dataset(val_dataset, block_size = config['block_size'])
    #     train_dataloader = DataLoader(train_dataset_down, batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers'])
    #     test_dataloader = DataLoader(test_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])
    #     val_dataloader = DataLoader(val_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])


    elif config['split'] == 'K_fold':
        print('K Fold spliting')
        dataset = pd.read_csv(config['downstream_dataset'], skiprows = 1, header = None)
        train_dataset, test_dataset = kfold_split(dataset, k = config['k_fold'], seed = config['seed'])
        
        #train_dataset_down = Downstream_dataset(train_dataset, block_size = config['block_size'])  
        train_dataset_down = [Downstream_dataset(train_dataset[i], block_size = config['block_size']) for i in range(config['k_fold'])]
        #test_dataset_down = Downstream_dataset(test_dataset, block_size = config['block_size']) 
        test_dataset_down = [Downstream_dataset(test_dataset[i], block_size = config['block_size']) for i in range(config['k_fold'])]
        #train_dataloader = DataLoader(train_dataset_down, batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers'])
        train_dataloader = [DataLoader(train_dataset_down[i], batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers']) for i in range(config['k_fold'])]
        #test_dataloader = DataLoader(test_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])
        test_dataloader = [DataLoader(test_dataset_down[i], batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers']) for i in range(config['k_fold'])]     
        
        polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')
        polycl.freeze_layers(polyBERT, layers_to_freeze = config["freeze_layers"])
        model = polycl.polycl_pred(encoder = polyBERT, pooler = config['pooler'], drop_ratio = config['dropout_ratio'])
        model.from_pretrained(config['pretrained_model_path'])
        loss_fn = nn.MSELoss()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        
        train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = [], [], [], []   
        best_r2 = 0.0
        for fold_num, dataloader in enumerate(train_dataloader):
            print('Fold %s/%s' % (fold_num + 1, config['k_fold']))
            # train_labels = []
            # for batch in dataloader:
            #     train_labels.append(batch['labels'])
            train_labels = [batch['labels'] for batch in dataloader]
            train_labels_tensor = torch.cat(train_labels, dim=0)
            # Instantiate the Normalizer
            normalizer = Normalizer(train_labels_tensor)
            optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

            torch.cuda.empty_cache()
            
            train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0
            
            count = 0
            
            current_test_dataloader = test_dataloader[fold_num]
            for epoch in range(config['n_epochs']):
                print("epoch: %s/%s" % (epoch + 1, config['n_epochs']))
                train(model, dataloader, device, optimizer, loss_fn, normalizer)
                train_loss, test_loss, r2_train, r2_test = eval(model, device, dataloader, current_test_dataloader, loss_fn, val_dataloader = None, normalizer = normalizer)

                if r2_test > best_test_r2:
                    best_train_r2 = r2_train
                    best_test_r2 = r2_test
                    train_loss_best = train_loss
                    test_loss_best = test_loss
                    count = 0
                else:
                    count += 1

                if r2_test > best_r2:
                    best_r2 = r2_test
                    model.save_model(path = config['best_model'])

                if count >= config['tolerance']:
                    print('Early stop')
                    if best_test_r2 == 0:
                        print('Bad performance with r^2 < 0')
                    break

            train_loss_avg.append(np.sqrt(train_loss_best))
            test_loss_avg.append(np.sqrt(test_loss_best))
            train_r2_avg.append(best_train_r2)
            test_r2_avg.append(best_test_r2)

        '''Average metrics'''
        train_rmse = np.mean(np.array(train_loss_avg))
        test_rmse = np.mean(np.array(test_loss_avg))
        train_r2 = np.mean(np.array(train_r2_avg))  
        test_r2 = np.mean(np.array(test_r2_avg))
        std_test_rmse = np.std(np.array(test_loss_avg))
        std_test_r2 = np.std(np.array(test_r2_avg))

        print('Train RMSE = ', train_rmse )
        print('Test RMSE = ', test_rmse)
        print('Train R2 = ', train_r2)
        print('Test R2 = ', test_r2)
        print('Std of Test RMSE = ', std_test_rmse)
        print('Std of Test R2 = ', std_test_r2)

    else:
        raise ValueError('Invalid split option')

    # train_dataset_down = Downstream_dataset(train_dataset, block_size = config['block_size'])
    # test_dataset_down = Downstream_dataset(test_dataset, block_size = config['block_size'])
    # train_dataloader = DataLoader(train_dataset_down, batch_size = config['batch_size'], shuffle = True, num_workers = config['num_workers'])
    # test_dataloader = DataLoader(test_dataset_down, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])
    # # if val_dataset:
    # #     val_dataloader = DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = False, num_workers = config['num_workers'])
    
    # polyBERT = AutoModel.from_pretrained('kuelumbus/polyBERT')
    # polycl.freeze_layers(polyBERT, layers_to_freeze = config["freeze_layers"])
    # model = polycl.polycl_pred(encoder = polyBERT, pooler = config['pooler'], drop_ratio = config['dropout_ratio'])
    # model.from_pretrained(config['pretrained_model_path'])
    # loss_fn = nn.MSELoss()
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # model.to(device)

    # train_labels = []
    # for batch in train_dataloader:
    #     train_labels.append(batch['labels'])
    # train_labels_tensor = torch.cat(train_labels, dim=0)
    # # Instantiate the Normalizer
    # normalizer = Normalizer(train_labels_tensor)
    # optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # torch.cuda.empty_cache()
    # best_r2 = 0.0
    # train_loss_best, test_loss_best, best_train_r2, best_test_r2 = 0.0, 0.0, 0.0, 0.0
    # train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = [], [], [], []
    # count = 0
    # for epoch in range(config['n_epochs']):
    #     print("epoch: %s/$s" % (epoch + 1, config['n_epochs']))
    #     train(model, train_dataloader, device, optimizer, loss_fn, normalizer)
    #     train_loss, test_loss, r2_train, r2_test = eval(model, device, train_dataloader, test_dataloader, loss_fn, val_dataloader = None, normalizer = normalizer)

    #     if r2_test > best_test_r2:
    #         best_train_r2 = r2_train
    #         best_test_r2 = r2_test
    #         train_loss_best = train_loss
    #         test_loss_best = test_loss
    #         count = 0
    #     else:
    #         count += 1

    #     if r2_test > best_r2:
    #         best_r2 = r2_test
    #         model.save_model(path = config['best_model'])

    #     if count >= config['tolerance']:
    #         print('Early stop')
    #         if best_test_r2 == 0:
    #             print('Bad performance with r^2 < 0')
    #         break

    # train_loss_avg.append(np.sqrt(train_loss_best))
    # test_loss_avg.append(np.sqrt(test_loss_best))
    # train_r2_avg.append(best_train_r2)
    # test_r2_avg.append(best_test_r2)

    # '''Average metrics'''
    # train_rmse = np.mean(np.array(train_loss_avg))
    # test_rmse = np.mean(np.array(test_loss_avg))
    # train_r2 = np.mean(np.array(train_r2_avg))
    # test_r2 = np.mean(np.array(test_r2_avg))
    # std_test_rmse = np.std(np.array(test_loss_avg))
    # std_test_r2 = np.std(np.array(test_r2_avg))

    # print('Train RMSE = ', train_rmse )
    # print('Test RMSE = ', test_rmse)
    # print('Train R2 = ', train_r2)
    # print('Test R2 = ', test_r2)
    # print('Std of Test RMSE = ', std_test_rmse)
    # print('Std of Test R2 = ', std_test_r2)

if __name__ == "__main__":
    main(config)
    
