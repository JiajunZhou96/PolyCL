import json
import random
import numpy as np
import torch

def read_txt(txt_name):
    
    with open(txt_name, 'r') as file:
        dataset = file.read().splitlines()
    
    return dataset

def get_config(path = None, print_dict = False):
    
    if path is None:
        file = './config.json'
    else:
        file = path

    f = open(file, 'r')
    line = f.read()
    config = json.loads(line)

    if print_dict:
        print(config)
    else:
        pass
    return config

def set_seed(seed):
    """
    Set seeds and enable deterministic settings for reproducibility.
    
    Parameters:
    - seed (int): Seed for random number generators.
    """
    # Setting seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Enabling determinism in cuDNN backend
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
