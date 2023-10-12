import json
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

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