import random
from rdkit import Chem
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer



# transfer smiles used for Transpoly to psmiles used for polyBERT
def to_psmiles(smiles):
    return smiles.replace("*", "[*]")

# transfer smiles used for polyBERT to psmiles used for Transpoly
def to_smiles(psmiles):
    return  psmiles.replace("[*]", "*")

# adapted from https://github.com/EBjerrum/SMILES-enumeration
def smiles_enumeration(smiles, random_seed = None):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    rng = np.random.default_rng(seed=random_seed)
    
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    rng.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    psmile = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
    
    return psmile.replace("*", "[*]")

def token_mask(smiles, rate = 0.1):

    tokens = list(smiles)
    num_to_mask = int(len(tokens) * rate)
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    for idx in mask_indices:
        tokens[idx] = "[MASK]"
    return "".join(tokens)

def insert(smiles,rate):
    
    return smiles

def token_drop(smiles, rate = 0.1):
    tokens = list(smiles)
    num_to_drop = int(len(tokens) * rate)
    drop_indices = random.sample(range(len(tokens)), num_to_drop)
    new_tokens = [token for idx, token in enumerate(tokens) if idx not in drop_indices]
    return "".join(new_tokens)

'''
def padding_to_max(encoded_input, max_length = 512):

    # Find the length of your sequences
    sequence_lengths = [len(seq) for seq in encoded_input['input_ids']]

    # Calculate how much padding is needed for each sequence
    padding_lengths = [max_length - seq_len for seq_len in sequence_lengths]

    # Pad your sequences
    padded_input_ids = torch.stack([torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)], dim=0) 
                                    for seq, pad_len in zip(encoded_input['input_ids'], padding_lengths)])

    padded_attention_mask = torch.stack([torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)], dim=0) 
                                        for seq, pad_len in zip(encoded_input['attention_mask'], padding_lengths)])

    padded_token_type_ids = torch.zeros((encoded_input['input_ids'].size()[0], 512), dtype=torch.long)
    
    return padded_input_ids, padded_attention_mask, padded_token_type_ids 
'''

class Construct_Dataset(Dataset):
    def __init__(self, smiles, labels=None, mode = "original"):
        self.smiles = smiles
        self.tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
        self.labels = labels
        self.mode = mode
    
    def __getitem__(self, idx):
        
        smile = self.smiles[idx]
        
        if self.mode == "original":  # 原版分子模式
            pass
        
        elif self.mode == "enumeration":
            smile = smiles_enumeration(smile)
        
        elif self.mode == "masking":
            smile = token_mask(smile, rate = 0.15)
            
        elif self.mode == "insert": #unfinished
            pass
        
        elif self.mode == "drop":
            smile =  token_drop(smile, rate = 0.15)
        
        # 这里的话 augmentation 在 tokenizer 的前面，所以不要担心会有丢失 special token 的问题
        encodings = self.tokenizer(smile, max_length= 512, padding="max_length", truncation=False, return_tensors='pt')
        #item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
        item = {key: val.squeeze(0) for key, val in encodings.items()}  # we are handling single sample, so need to modify to this form
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.smiles)
    
class Normalizer(object):
    def __init__ (self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
class Downstream_dataset(Dataset):
    def __init__(self, data, block_size):
        self.tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')
        self.data = data
        self.block_size = block_size
        
    def __len__(self):
        self.len = len(self.data)
        return self.len
    
    def __getitem__(self, i):
        row = self.data.iloc[i]
        smiles = row[0]
        labels = row[1]
        
        encodings = self.tokenizer(smiles, 
                                   truncation = True, 
                                   max_length = self.block_size, 
                                   padding = 'max_length', 
                                   return_tensors = 'pt')    
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype = torch.float)
            }