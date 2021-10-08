import torch
import torch.nn as nn
import csv
import time
import logging

def get_ctime() :
    return time.strftime("%y-%m-%d %H:%M:%S", time.gmtime(time.time()+32400))

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

def init_model (model) :
    for param in model.parameters() :
        if param.dim() == 1 :
            continue
        else :
            nn.init.xavier_normal_(param)

def set_device(gpus) -> torch.device :
    if gpus == 0 :
        return torch.device('cpu')
    elif torch.cuda.device_count() == 0 :
        logging.info("Warning: No Available GPU, CPU is used")
        return torch.device('cpu')
    else :
        return torch.device('cuda')

