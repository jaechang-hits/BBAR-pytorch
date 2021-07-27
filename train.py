import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import random
import sys
import gc

from fcp import FCP
from cond_module import Cond_Module
from ns_module import NS_Trainer
from dataset import FCPDataset
from utils import common, brics

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# set parameters
gpus = 1    # I don't implement multi-gpu
num_workers = 8
lr = 0.001
batch_size = 128
batch_size_val = 256
datapoint_per_epoch = 4000000
max_atoms = 50
max_epoch = 100
n_sample = 10
train_file = 'data/train.csv'
train_freq_file = 'data/train_freq.npy'
val_file = 'data/val.csv'
balanced = True
library_file = 'data/library.csv'
library_npz_file = 'data/library.npz'
db_file = 'data/malt_logp_qed.db'
model_save = True
target = ['MolLogP']
loss_fn = 'mean'

label_smoothing = float(sys.argv[1])
dropout = float(sys.argv[2])
save_dir = sys.argv[3]

if gpus == 1 :
    device ='cuda:0'
else :
    device = 'cpu'

if not os.path.isdir(save_dir) :
    os.mkdir(save_dir)

if len(target) > 0 :
    cond_module = Cond_Module(db_file, target)
    cond_scale = cond_module.scale
else :
    cond_module = None
    cond_scale = {}

train_ds = FCPDataset(train_file, cond_module, max_atoms) 
val_ds = FCPDataset(val_file, cond_module, max_atoms) 

if balanced :
    freq = np.load(train_freq_file)
    weight = 1. / freq
    weight = torch.from_numpy(weight).double()
    sampler = WeightedRandomSampler(weight, datapoint_per_epoch)
    train_dl = DataLoader(train_ds, batch_size, num_workers=num_workers, sampler=sampler)
    gc.collect()
else :
    train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers=num_workers)
    n_class = len(train_ds)

val_dl = DataLoader(val_ds, batch_size_val, shuffle = False, num_workers=num_workers)

n_train = len(train_ds)
n_val = len(val_ds)

model = FCP(cond_scale, dropout = dropout)
common.init_model(model)
trainer = NS_Trainer(model, library_npz_file, n_sample, 3/4, label_smoothing, device)
print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('num of train data:', n_train)
print('num of datapoint per epoch:', datapoint_per_epoch)
print('num of val data:', n_val)

for epoch in range(max_epoch) :
    st = time.time()
    # Train
    model.train()
    model.zero_grad()
    tloss_list = []
    fploss_list = []
    fnloss_list = []
    iloss_list = []
    for i_batch, (h_in, adj_in, cond, y_fid, y_idx) in enumerate(train_dl) :
        h_in = h_in.to(device).float()
        adj_in = adj_in.to(device).bool()
        cond = cond.to(device).float()
        y_fid = y_fid.to(device).long()
        y_idx = y_idx.to(device).long()

        term_loss, fid_ploss, fid_nloss, idx_loss = trainer(h_in, adj_in, cond, y_fid, y_idx)
        if loss_fn =='mean' :
            fid_loss = (fid_ploss + fid_nloss / n_sample)
        else :
            fid_loss = (fid_ploss + fid_nloss)
        loss = term_loss + fid_loss + idx_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
        tloss_list.append(term_loss.detach().cpu())
        if isinstance(idx_loss, float) :
            continue        # just termination loss
        fploss_list.append(fid_ploss.detach().cpu())
        fnloss_list.append((fid_nloss/n_sample).detach().cpu())
        iloss_list.append(idx_loss.detach().cpu())

    model.eval()
    if model_save :
        trainer.model_save_gv_lib()
        save_file = os.path.join(save_dir, f'save{epoch}.pt')
        torch.save(model, save_file)

    tloss = np.mean(np.array(tloss_list))
    fploss = np.mean(np.array(fploss_list))
    fnloss = np.mean(np.array(fnloss_list))
    iloss = np.mean(np.array(iloss_list))

    ctime = common.get_ctime()
    end = time.time()
    print(
        f'[{ctime}]\t'
        f'Epoch {epoch} \t'
        'train\t'
        f'tloss: {tloss:.3f}\t'
        f'iloss: {iloss:.3f}\t'
        f'ploss: {fploss:.3f}\t'
        f'nloss: {fnloss:.3f}\t'
        f'time: {end-st:.3f}'
    )

    # Validation
    st = time.time()
    model.eval()
    tloss_list = []
    fploss_list = []
    fnloss_list = []
    iloss_list = []
    with torch.no_grad() :
        for i_batch, (h_in, adj_in, cond, y_fid, y_idx) in enumerate(val_dl) :
            h_in = h_in.to(device).float()
            adj_in = adj_in.to(device).bool()
            cond = cond.to(device).float()
            y_fid = y_fid.to(device).long()
            y_idx = y_idx.to(device).long()

            term_loss, fid_ploss, fid_nloss, idx_loss = trainer(h_in, adj_in, cond, y_fid, y_idx, False)
            if loss_fn =='mean' :
                fid_loss = (fid_ploss + fid_nloss / n_sample)
            else :
                fid_loss = (fid_ploss + fid_nloss)
            tloss_list.append(term_loss.detach().cpu())
            if isinstance(idx_loss, float) :
                continue        # just termination loss
            iloss_list.append(idx_loss.detach().cpu())
            fploss_list.append(fid_ploss.detach().cpu())
            fnloss_list.append((fid_nloss/n_sample).detach().cpu())

    tloss = np.mean(np.array(tloss_list))
    iloss = np.mean(np.array(iloss_list))
    fploss = np.mean(np.array(fploss_list))
    fnloss = np.mean(np.array(fnloss_list))
    
    ctime = common.get_ctime()
    end = time.time()
    print(
        f'[{ctime}]\t'
        f'Epoch {epoch} \t'
        'val  \t'
        f'tloss: {tloss:.3f}\t'
        f'iloss: {iloss:.3f}\t'
        f'ploss: {fploss:.3f}\t'
        f'nloss: {fnloss:.3f}\t'
        f'time: {end-st:.3f}\n'
    )

    trainer.model_remove_gv_lib()
