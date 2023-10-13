import os
import torch
import torch.nn as nn


import utils
config = utils.get_config(print_dict = False)
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0, :]


'''
def max_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    max_over_time = torch.max(token_embeddings, 1)[0]
    return max_over_time

def min_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = 1e9  # Set padding tokens to large positive value
    min_over_time = torch.min(token_embeddings, 1)[0]
    return min_over_time
'''

def first_last_pooling(model_output, attention_mask):

    first_hidden = model_output.hidden_states[1]
    last_hidden = model_output.hidden_states[-1]
    pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
    
    return pooled_result

# contrastive learning based on polyCL
class polyCL(nn.Module):

    def __init__(self, encoder, pooler):
        super(polyCL, self).__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.projection_head = nn.Sequential(nn.Linear(600, 256),   # 这里第一个向量应该是 embedding size
                                            nn.ReLU(inplace=True), 
                                            nn.Linear(256, 128))

    def forward(self, data):  # dataloader
        
        ## if self.pooler: # or "mean" or " max"
        model_output = self.encoder(input_ids = data["input_ids"], attention_mask = data["attention_mask"], token_type_ids = data["token_type_ids"])
        
        # 如果不进行 pooling 的话，这个 model_output 就只能通过 rnn 类的时间序列模型来进行处理了
        if self.pooler == "mean":
            rep = mean_pooling(model_output, model_output['attention_mask'])
        
        #elif self.pooler == "max":
        #    rep = max_pooling(model_output, model_output['attention_mask'])
        
        #elif self.pooler == "min":
        #    rep = min_pooling(model_output, model_output['attention_mask'])
        
        elif self.pooler == "cls":
            rep = cls_pooling(model_output)
        
        elif self.pooler == "first_last":
            rep = first_last_pooling(model_output, model_output['attention_mask'])
        
        out = self.projection_head(rep)
        
        return rep, out
    
    def save_model(self, path = None):
        
        if path is None:
            path = os.path.join(os.getcwd(), 'model')
        else:
            path = os.path.join(os.getcwd(), path)
        
        if os.path.exists(path) == True:
            pass
            print('Path already existed.')
        else:
            os.mkdir(path)
            
        print('Path created.')
        
        torch.save(self.state_dict(), path + '/polycl_model.pth')
        
    def load_pretrained(self):
        
        pass
        

import torch
import numpy as np

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
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
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
    
def freeze_layers(model, layers_to_freeze = False):

    #Freeze layers
    if layers_to_freeze is False:
        pass
    
    elif isinstance(layers_to_freeze, int):
        for i in range(layers_to_freeze):
            for param in model.encoder.layer[i].parameters():
                param.requires_grad = False

    elif isinstance(layers_to_freeze, list):
        for idx in layers_to_freeze:
            for param in model.encoder.layer[idx].parameters():
                param.requires_grad = False
    
    else:
        raise ValueError("Input layers_to_freeze should be an int or a list of int.")
