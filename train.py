import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import random
import sys

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
max_atoms = 50
max_epoch = 50
n_sample = 10
train_file = 'data/train_uniq.csv'
balanced = False
library_file = 'data/library.csv'
library_npz_file = 'data/library.npz'
db_file = 'data/malt_logp_qed.db'
save_dir = 'save_logp'
model_save = True
target = ['MolLogP']
loss_fn = 'sum'

if gpus == 1 :
    device ='cuda:0'
else :
    device = 'cpu'

if not os.path.isdir(save_dir) :
    os.mkdir(save_dir)

library = brics.BRICSLibrary(library_file, True)

if len(target) > 0 :
    cond_module = Cond_Module(db_file, target)
    cond_scale = cond_module.scale
else :
    cond_module = None
    cond_scale = {}

train_ds = FCPDataset(train_file, cond_module, library, max_atoms) 

if balanced :
    class_sample_count = np.array([len(np.where(train_ds.frag1==t)[0]) for t in train_ds.frag1])
    weight = 1. / class_sample_count
    weight = torch.from_numpy(weight).double()
    n_class = len(set(train_ds.frag1.tolist()))
    sampler = WeightedRandomSampler(weight, n_class)
    train_dl = DataLoader(train_ds, batch_size, num_workers=num_workers, sampler=sampler)
else :
    train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers=num_workers)
    n_class = len(train_ds)

n_train = len(train_ds)

model = FCP(library, cond_scale)
common.init_model(model)
trainer = NS_Trainer(library_npz_file, n_sample, 3/4, model, device)
print(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('num of train data:', n_train)
print('num of datapoint per epoch:', n_class)

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
            fid_loss = -(fid_ploss + fid_nloss / n_sample)
        else :
            fid_loss = -(fid_ploss + fid_nloss)
        loss = term_loss + fid_loss + idx_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.zero_grad()
        tloss_list.append(term_loss.detach().cpu())
        fploss_list.append(-fid_ploss.detach().cpu())
        fnloss_list.append((-fid_nloss/n_sample).detach().cpu())
        iloss_list.append(idx_loss.detach().cpu())

    if model_save :
        trainer.model_save_gv_lib()
        save_file = os.path.join(save_dir, f'save{epoch}.pt')
        torch.save(model, save_file)
        trainer.model_remove_gv_lib()

    tloss = np.mean(np.array(tloss_list))
    fploss = np.mean(np.array(fploss_list))
    fnloss = np.mean(np.array(fnloss_list))
    iloss = np.mean(np.array(iloss_list))
    end = time.time()
    ctime = common.get_ctime()
    print(
        f'[{ctime}]\t'
        f'Epoch {epoch} \t'
        f'tloss: {tloss:.3f}\t'
        f'ploss: {fploss:.3f}\t'
        f'nloss: {fnloss:.3f}\t'
        f'iloss: {iloss:.3f}\t'
        f'time: {end-st:.3f}'
    )
    continue
    # Validation
    # Pass

