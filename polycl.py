import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import utils
config = utils.get_config(print_dict = False)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0, :]

def first_last_pooling(model_output, attention_mask):

    first_hidden = model_output.hidden_states[1]
    last_hidden = model_output.hidden_states[-1]
    pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    
    return pooled_result

class polyCL(nn.Module):

    def __init__(self, encoder, pooler):
        super(polyCL, self).__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.projection_head = nn.Sequential(nn.Linear(600, 256),
                                            nn.ReLU(inplace=True), 
                                            nn.Linear(256, 128))

    def forward(self, data):  # dataloader
        if "token_type_ids" in data:
            model_output = self.encoder(input_ids = data["input_ids"], attention_mask = data["attention_mask"], token_type_ids = data["token_type_ids"])
        else:
            model_output = self.encoder(input_ids = data["input_ids"], attention_mask = data["attention_mask"])
        
        if self.pooler == "cls":
            rep = cls_pooling(model_output)
        
        elif self.pooler == "first_last":
            rep = first_last_pooling(model_output, model_output['attention_mask'])
        
        out = self.projection_head(rep)
        
        return rep, out

    def save_model(self, path = None):
        
        if isinstance(self, nn.DataParallel):
            torch.save(self.module.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)
    
    def save_checkpoint(self, file_path, optimizer=None, scheduler=None, scaler=None, epoch=None):
        """Save the model checkpoint."""
        model_state = self.module.state_dict() if isinstance(self, nn.DataParallel) else self.state_dict()
        checkpoint = {
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'epoch': epoch
        }
        torch.save(checkpoint, file_path)
    
    def load_checkpoint(self, file_path, optimizer=None, scheduler=None, scaler=None):
        """Load the model checkpoint."""
        checkpoint = torch.load(file_path)
        model_state = self.module if isinstance(self, nn.DataParallel) else self
        model_state.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'epoch' in checkpoint:
            return checkpoint['epoch']


#https://github.com/yuyangw/MolCLR/blob/master/molclr.py
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

def set_dropout(config, dropout):
    if dropout == True:
        pass
    elif dropout == False:
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
    
    return config

class polycl_pred(torch.nn.Module):
    def __init__(self, PretrainedModel, drop_ratio = 0, hidden_size = 256, activation_func = nn.ReLU(inplace = True)):
        super(polycl_pred, self).__init__()

        self.PretrainedModel = deepcopy(PretrainedModel)
        
        if hasattr(self.PretrainedModel, 'config'):
            input_size = self.PretrainedModel.config.hidden_size
        elif hasattr(self.PretrainedModel, 'encoder') and hasattr(self.PretrainedModel.encoder, 'config'):
            input_size = self.PretrainedModel.encoder.config.hidden_size
        else:
            raise ValueError("Invalid pre-trained model or missing configuration")
        
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            activation_func,
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_size, 1)
        )
    
    def from_pretrained(self, model_file):
        self.PretrainedModel.load_state_dict(torch.load(model_file, map_location = 'cpu'))
        self.PretrainedModel.to(device)

    def forward(self, input_ids = None, attention_mask = None, data = None):
        if data is None:
            data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
            }
            output = self.PretrainedModel(input_ids = input_ids, attention_mask = attention_mask)
        else:
            output = self.PretrainedModel(data)

        if isinstance(output, tuple):
            rep, _ = output
        else:
            rep  = output.last_hidden_state[:, 0, :] # cls pooling
        regression = self.regressor(rep)

        return regression

# For TransPolymer downstream evaluation
class Downstream_regression(nn.Module):
    def __init__(self, PretrainedModel, drop_ratio = 0, activation_func = nn.ReLU(inplace = True)):
        super(Downstream_regression, self).__init__()

        self.PretrainedModel = deepcopy(PretrainedModel)

        self.regressor = nn.Sequential(
            nn.Linear(self.PretrainedModel.config.hidden_size, 256),
            activation_func,
            nn.Dropout(drop_ratio),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids = input_ids, attention_mask = attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.regressor(logits)

        return output
