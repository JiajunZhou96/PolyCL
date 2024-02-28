import random
from rdkit import Chem
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer



# to psmiles(polyBERT)
def to_psmiles(smiles):
    return smiles.replace("*", "[*]")

# to smiles
def to_smiles(psmiles):
    return  psmiles.replace("[*]", "*")

# adapted from https://github.com/EBjerrum/SMILES-enumeration
def smiles_enumeration(smiles, random_seed = None):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    rng = np.random.default_rng(seed=random_seed)
    m = Chem.MolFromSmiles(smiles)
    if m is None or m.GetNumAtoms() == 0:
        return smiles
    try:
        Chem.SanitizeMol(m)
        ans = list(range(m.GetNumAtoms()))
        rng.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        psmile = Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)
        return psmile.replace("*", "[*]")
    except Exception as e:
        return smiles

def token_mask(smiles, rate = 0.1):

    tokens = list(smiles)
    num_to_mask = int(len(tokens) * rate)
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    for idx in mask_indices:
        tokens[idx] = "[MASK]"
    return "".join(tokens)

def token_drop(smiles, rate = 0.1):
    tokens = list(smiles)
    num_to_drop = int(len(tokens) * rate)
    drop_indices = random.sample(range(len(tokens)), num_to_drop)
    new_tokens = [token for idx, token in enumerate(tokens) if idx not in drop_indices]
    return "".join(new_tokens)

class Construct_Dataset(Dataset):
    def __init__(self, smiles, labels=None, mode = "original", tokenizer = AutoTokenizer.from_pretrained('kuelumbus/polyBERT')):
        self.smiles = smiles
        self.tokenizer = tokenizer
        self.labels = labels
        self.mode = mode
    
    def __getitem__(self, idx):
        
        smile = self.smiles[idx]
        
        if self.mode == "original":
            pass
        
        elif self.mode == "enumeration":
            smile = smiles_enumeration(smile)
        
        elif self.mode == "masking":
            smile = token_mask(smile, rate = 0.15)
        
        elif self.mode == "drop":
            smile =  token_drop(smile, rate = 0.15)
        
        encodings = self.tokenizer(smile, max_length= 512, padding="max_length", truncation=False, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.smiles)


class Downstream_dataset(Dataset):
    def __init__(self, data, block_size, tokenizer):
        self.tokenizer = tokenizer
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
        
