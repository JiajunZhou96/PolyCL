import json
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from itertools imoprt compress
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold
from collections import defaultdict

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

def get_scheduler(config, optimizer):
    config = get_config(print_dict = False)
    scheduler_config = config["scheduler"]
    scheduler_type = scheduler_config["type"]
    optional_params = scheduler_config.get("optional_params", {})

    if scheduler_type == "StepLR":
        return lr_scheduler.StepLR(optimizer, 
                                   step_size = scheduler_config["step_size"],
                                   gamma = scheduler_config["gamma"],
                                   **optional_params)
    elif scheduler_type == "CyclicLR":
        return lr_scheduler.CyclicLR(optimizer,
                                     base_lr = config["lr"],
                                     max_lr = scheduler_config.get(["max_lr"], 0.01),
                                     step_size_up = scheduler_config["step_size"],
                                     **optional_params)
    elif scheduler_type == "LinearLR":
        return lr_scheduler.LinearLR(optimizer,
                                    start_factor = scheduler_config["start_factor"],
                                    total_iters = scheduler_config["total_iters"],
                                    **optional_params)

'''https://github.com/ssnl/align_uniform/blob/master/examples/stl10/util.py'''
class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


'''Split options'''
'''scaffold spliter'''
def generate_scaffold(mol, include_chirality = False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles = mol, includeChirality = include_chirality)
    return scaffold

def scaffold_split(dataset, mol_list, train_frac = 0.8, test_frac = 0.1, val_frac = 0.1, include_chirality = False):

    np.testing.assert_almost_equal(train_frac + test_frac + val_frac, 1.0)

    all_scaffolds = defaultdict(list)
    for idx, mol in enumerate(mol_list):
        scaffold = generate_scaffold(mol, include_chirality = True)
        all_scaffolds[scaffold].append(idx)

    #Sort the scaffolds by the number of molecules they correspond to
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffolds_sorted = [scaffold_set for (scaffold, scaffold_set) in sorted(all_scaffolds.items(), key = lambda x: (len(x[1]), x[1][0]), reverse = True)]
    #scaffolds_sorted = sorted(scaffolds.values(), key = len, reverse = True)

    train_cutoff = train_frac * len(mol_list)
    val_cutoff = (train_frac + val_frac) * len(mol_list)
    train_idx, test_idx, val_idx = [], [], []

    for scaffolds_sorted in all_scaffolds_sorted:
        if len(train_idx) + len(scaffolds_sorted) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffolds_sorted) > val_cutoff:
                test_idx.extend(scaffolds_sorted)
            else:
                val_idx.extend(scaffolds_sorted)
        else:
            train_idx.extend(scaffolds_sorted)

    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(test_idx).intersection(set(val_idx))) == 0

    # train_dataset = dataset[torch.tensor(train_idx)]
    # test_dataset = dataset[torch.tensor(test_idx)]
    # val_dataset = dataset[torch.tensor(val_idx)]

    train_dataset = dataset.iloc[train_idx]
    val_dataset = dataset.iloc[val_idx]
    test_dataset = dataset.iloc[test_idx]

    return train_dataset, test_dataset, val_dataset

'''K_fold splitter'''
def kfold_split(data, k=5, seed=1):
    splits = KFold(n_splits=k, shuffle=True, random_state=seed)
    #data_numeric = data.select_dtypes(include = [np.number])
    train_dataset = []
    test_dataset = []
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data.shape[0]))):
        print('Fold {}'.format(fold + 1))
        
        # train_data = torch.tensor(data.iloc[train_idx, 1].values, dtype = torch.float32)
        # test_data = torch.tensor(data.iloc[val_idx, 1].values, dtype = torch.float32)

        train_data = data.loc[train_idx, :].reset_index(drop=True)
        test_data = data.loc[val_idx, :].reset_index(drop=True)
        
        train_dataset.append(train_data)
        test_dataset.append(test_data)
    return train_dataset, test_dataset
    
'''Random splitter (array based)'''
def random_split(dataset, mol_list = None, train_frac = 0.8, test_frac = 0.1, val_frac = 0.1):

    np.testing.assert_almost_equal(train_frac + test_frac + val_frac, 1.0)

    num_mols = len(dataset)
    random.seed(72)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(train_frac * num_mols)]
    val_idx = all_idx[int(train_frac * num_mols):int(val_frac * num_mols) + int(train_frac * num_mols)]
    test_idx = all_idx[int(val_frac * num_mols) + int(train_frac * num_mols):]

    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == num_mols

    # train_dataset = torch.tensor(dataset.iloc[train_idx, 1].values, dtype=torch.float32)
    # val_dataset = torch.tensor(dataset.iloc[val_idx, 1].values, dtype=torch.float32)
    # test_dataset = torch.tensor(dataset.iloc[test_idx, 1].values, dtype=torch.float32)

    train_dataset = dataset.iloc[train_idx]
    val_dataset = dataset.iloc[val_idx]
    test_dataset = dataset.iloc[test_idx]

    if not mol_list:
        return train_dataset, val_dataset, test_dataset
        
    else:
        train_smiles = [mol_list[i] for i in train_idx]
        val_smiles = [mol_list[i] for i in val_idx]
        test_smiles = [mol_list[i] for i in test_idx]

        return train_dataset, test_dataset, val_dataset, (train_smiles, val_smiles, test_smiles)

