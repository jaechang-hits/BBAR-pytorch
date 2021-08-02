import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
import random
import sys
import gc
import logging

from fcp import FCP
from cond_module import Cond_Module
from ns_module import NS_Trainer
from dataset import FCPDataset
from utils import common, brics
from utils.hydra_runner import hydra_runner
from utils.exp_manager import exp_manager

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

@hydra_runner(config_path='conf', config_name='train')
def main(cfg) : 
    train_cfg = cfg.train
    ns_cfg = cfg.ns_trainer
    cond_cfg = cfg.condition
    data_cfg = cfg.data
    cfg, save_dir = exp_manager(cfg, cfg.exp_dir)

    if train_cfg.gpus == 1 :
        device ='cuda:0'
    else :
        device = 'cpu'

    if len(cond_cfg.descriptors) > 0 :
        cond_module = Cond_Module(cond_cfg.db_file, cond_cfg.descriptors)
        cond_scale = cond_module.scale
    else :
        cond_module = None
        cond_scale = {}

    train_ds = FCPDataset(data_cfg.train.data_path, cond_module, data_cfg.train.max_atoms) 
    val_ds = FCPDataset(data_cfg.val.data_path, cond_module, data_cfg.train.max_atoms) 

    weight = torch.from_numpy(np.load(data_cfg.train.sampler.weight_path)).double()
    sampler = WeightedRandomSampler(weight, data_cfg.train.sampler.n_sample)
    train_dl = DataLoader(train_ds, data_cfg.train.batch_size, num_workers=data_cfg.train.num_workers, sampler=sampler)
    val_dl = DataLoader(val_ds, data_cfg.val.batch_size, num_workers=data_cfg.val.num_workers)

    n_train = len(train_ds)
    n_val = len(val_ds)

    model = FCP(cond_scale)
    common.init_model(model)
    trainer = NS_Trainer(model, ns_cfg.library_feature_path, ns_cfg.n_sample, ns_cfg.alpha, device)
    logging.info(f"number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    logging.info(f'num of train data: {n_train}')
    logging.info(f'num of datapoint per epoch: {data_cfg.train.sampler.n_sample}')
    logging.info(f'num of val data: {n_val}\n')

    for epoch in range(train_cfg.max_epoch) :
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
            fid_loss = fid_ploss + fid_nloss
            loss = term_loss + fid_loss + idx_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
            tloss_list.append(term_loss.detach().cpu())
            if isinstance(idx_loss, float) :
                continue        # just termination loss
            fploss_list.append(fid_ploss.detach().cpu())
            fnloss_list.append(fid_nloss.detach().cpu())
            iloss_list.append(idx_loss.detach().cpu())

        tloss = np.mean(np.array(tloss_list))
        fploss = np.mean(np.array(fploss_list))
        fnloss = np.mean(np.array(fnloss_list))
        iloss = np.mean(np.array(iloss_list))

        ctime = common.get_ctime()
        end = time.time()
        logging.info(
            f'[{ctime}]\t'
            f'Epoch {epoch} \t'
            'train\t'
            f'tloss: {tloss:.3f}\t'
            f'iloss: {iloss:.3f}\t'
            f'ploss: {fploss:.3f}\t'
            f'nloss: {fnloss:.3f}\t'
            f'time: {end-st:.3f}'
        )
        
        model.eval()
        trainer.model_save_gv_lib()
        save_file = os.path.join(save_dir, f'save{epoch}.pt')
        torch.save(model, save_file)

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
                tloss_list.append(term_loss.detach().cpu())
                if isinstance(idx_loss, float) :
                    continue        # just termination loss
                iloss_list.append(idx_loss.detach().cpu())
                fploss_list.append(fid_ploss.detach().cpu())
                fnloss_list.append(fid_nloss.detach().cpu())

        tloss = np.mean(np.array(tloss_list))
        iloss = np.mean(np.array(iloss_list))
        fploss = np.mean(np.array(fploss_list))
        fnloss = np.mean(np.array(fnloss_list))
        
        ctime = common.get_ctime()
        end = time.time()
        logging.info(
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

if __name__ == '__main__' :
    main()
