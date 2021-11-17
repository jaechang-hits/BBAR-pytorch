import torch
import torch.nn as nn
import numpy as np
import random
import csv
import time
import logging

def get_ctime(timezone = None) :
    if timezone is None :
        return time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))
    else :
        return time.strftime("%y-%m-%d %H:%M:%S", time.gmtime(time.time()+timezone*3600))

def load_csv(csvfile: str) -> list :
    with open(csvfile) as f :
        lines = list(csv.reader(f))
#        lines = csv.reader(f)
    return lines

def load_txt(txtfile: str) -> list :
    with open(txtfile) as f :
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines

def set_device(gpus) -> torch.device :
    if gpus == 0 :
        return torch.device('cpu')
    elif torch.cuda.device_count() == 0 :
        logging.info("Warning: No Available GPU, CPU is used")
        return torch.device('cpu')
    else :
        return torch.device('cuda')

def set_seed(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
