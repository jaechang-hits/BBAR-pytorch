import torch.nn as nn
import csv
import time

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
