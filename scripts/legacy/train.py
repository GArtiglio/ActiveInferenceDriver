import argparse
import os
import pickle
import time
import json
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from src.legacy.ebirl import EBIRL

class DriverDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.eps_ids = self.df["episode"].unique()

    def __len__(self):
        return len(self.eps_ids)

    def __getitem__(self, idx):
        df_drive = self.df.loc[self.df["episode"] == self.eps_ids[idx]].iloc[:30]
        obs = torch.from_numpy(df_drive["obs"].to_numpy()).unsqueeze(-1).to(torch.float32)
        act = torch.from_numpy(df_drive["act"].to_numpy()).long()
        
        mask = torch.ones_like(act)
        if "rt" in df_drive.columns:
            rt = df_drive["rt"].iloc[0] 
            mask[rt+2:] = 0
        return obs, act, mask

def train_test_split(
    df, batch_size, train_ratio=1., iterative_inference=False, seed=0
    ):
    assert train_ratio <= 1.
    eps_ids = df["episode"].unique()
    random.Random(seed).shuffle(eps_ids)
    num_eps = len(eps_ids)
    
    shuffle = True
    if iterative_inference:
        batch_size = num_eps
        train_ratio = 1
        shuffle = False
    num_train = np.floor(train_ratio * num_eps).astype(int)
    
    print(
        "Splitting data with train_ratio: {}, batch_size: {},"
        "shuffle: {}, seed: {}".format(
            train_ratio, batch_size, shuffle, seed
        )
    )
    
    df_train = df.loc[df["episode"].isin(eps_ids[:num_train])]
    df_test = df.loc[df["episode"].isin(eps_ids[num_train:])]
    
    train_loader = DataLoader(
        DriverDataset(df_train), batch_size=batch_size, shuffle=shuffle
    )
    test_loader = DataLoader(
        DriverDataset(df_test), batch_size=batch_size, shuffle=shuffle
    ) if train_ratio < 1 else None
            
    return train_loader, test_loader

def init_model(arglist):
    model = EBIRL(
        arglist.state_dim,
        arglist.obs_dim,
        arglist.act_dim,
        arglist.horizon,
        arglist.precision,        
        arglist.rnn_hidden_dim,
        arglist.rnn_stack_dim,
        arglist.rnn_dropout,
        rnn_bidirectional=arglist.rnn_bidirectional,
        # prior settings
        prior_dist=arglist.prior_dist,
        num_components=arglist.num_components,
        # post settings
        post_dist=arglist.post_dist,
        num_flows=arglist.num_flows,
        iterative=arglist.iterative,
        batch_size=arglist.vi_batch_size,
        init_uniform=arglist.init_uniform,
        logvar_offset=arglist.logvar_offset,
        # learning settings
        prior_lr=arglist.prior_lr,
        prior_decay=arglist.prior_decay,
        prior_schedule_steps=arglist.prior_schedule_steps,
        prior_schedule_gamma=arglist.prior_schedule_gamma,
        vi_lr=arglist.vi_lr,
        vi_decay=arglist.vi_decay,
        vi_schedule_steps=arglist.vi_schedule_steps,
        vi_schedule_gamma=arglist.vi_schedule_gamma,
        mc_samples=arglist.mc_samples,
        obs_penalty=arglist.obs_penalty,
        kl_penalty=arglist.kl_penalty,
        kl_anneal_steps=arglist.kl_anneal_steps,
        kl_const_steps=arglist.kl_const_steps,
        kl_cycles=arglist.kl_cycles,
        grad_clip=arglist.grad_clip,
        seed=arglist.seed
    )
    return model

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../dat/processed", help="data path, default=../dat/processed")
    parser.add_argument("--exp_path", type=str, default="../exp/ebirl", help="experiment path, default=../exp/ebirl")
    parser.add_argument("--train_ratio", type=float, default=1., help="train split ratio, default=1")
    # agent args
    parser.add_argument("--state_dim", type=int, default=2, help="") 
    parser.add_argument("--obs_dim", type=int, default=1, help="")
    parser.add_argument("--act_dim", type=int, default=2, help="")
    parser.add_argument("--horizon", type=int, default=35, help="agent max planning horizon, default=35")
    parser.add_argument("--precision", type=int, default=1, help="fit agent precision, 0 or 1") 
    # prior args 
    parser.add_argument("--prior_dist", type=str, default="mvn", help="prior dist type: [diag, mvn, gmm], default=mvn")
    parser.add_argument("--num_components", type=int, default=3, help="number of gmm components, default=3")
    parser.add_argument("--prior_lr", type=float, default=0.01, help="prior learning rate, default=0.01")
    parser.add_argument("--prior_decay", type=float, default=0, help="prior weight decay, default=0")
    parser.add_argument("--prior_schedule_steps", type=int, default=1000, help="prior lr scheduler step size, default=1000")
    parser.add_argument("--prior_schedule_gamma", type=float, default=0.8, help="prior lr scheduler decay rate, default=0.8")
    # vi args
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="rnn encoder hidden dim")
    parser.add_argument("--rnn_stack_dim", type=int, default=2, help="rnn encoder layers")
    parser.add_argument("--rnn_dropout", type=float, default=0, help="rnn encoder dropout")
    parser.add_argument("--rnn_bidirectional", type=bool_, default=True, help="rnn encoder bidirectional")
    parser.add_argument("--post_dist", type=str, default="diag", help="posterior dist type: [diag, mvn, norm_flow], default=diag")
    parser.add_argument("--num_flows", type=int, default=26, help="number of flow layers for encoder, default=26")
    parser.add_argument("--iterative", type=bool_, default=True, help="whether to use iterative inference, default=True")
    parser.add_argument("--init_uniform", type=bool_, default=True, help="whether to init encoder params uniformly, default=True")
    parser.add_argument("--logvar_offset", type=float, default=2., help="encoder logvar offset to init small, default=2")
    parser.add_argument("--vi_batch_size", type=int, default=32, help="variational inference batch size, default=32")
    parser.add_argument("--vi_lr", type=float, default=0.01, help="variational inference learning rate, default=0.01")
    parser.add_argument("--vi_decay", type=float, default=0, help="variational inference weight decay, default=0")
    parser.add_argument("--vi_schedule_steps", type=int, default=1000, help="variational inference lr scheduler step size, default=1000")
    parser.add_argument("--vi_schedule_gamma", type=float, default=0.8, help="variational inference lr scheduler decay rate, default=0.8")
    # train args
    parser.add_argument("--mc_samples", type=int, default=5, help="elbo mc samples, default=3")
    parser.add_argument("--obs_penalty", type=float, default=0.2, help="observation likelihood penalty, default=0.2")
    parser.add_argument("--kl_penalty", type=float, default=0, help="initial kl penalty, default=1e-5")
    parser.add_argument("--kl_anneal_steps", type=int, default=5000, help="kl annealing steps, default=5000")
    parser.add_argument("--kl_const_steps", type=int, default=1000, help="constant kl=1 steps, default=1000")
    parser.add_argument("--kl_cycles", type=int, default=1, help="initial kl penalty, default=1")
    parser.add_argument("--grad_clip", type=float, default=20., help="gradient clipping, default=20")
    parser.add_argument("--verbose", type=int, default=10, help="training verbose interval, default=10")
    parser.add_argument("--epochs", type=int, default=10000, help="training epochs, default=10000")
    parser.add_argument("--seed", type=int, default=0, help="random seed, default=0")
    parser.add_argument("--save", type=bool_, default=True, help="save results, default=True")
    parser.add_argument("--debug", type=bool_, default=False, help="debug mode, default=False")
    arglist = parser.parse_args()
    return arglist

def run_epoch(model, loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    epoch_stats = []
    num_samples = []
    for i, data_batch in enumerate(loader):        
        if train:
            stats_dict = model.take_gradient_step(data_batch)
        else:
            with torch.no_grad():
                _, stats_dict = model.evaluate(data_batch)

        epoch_stats.append(stats_dict)
        num_samples.append(len(data_batch[0]))
    
    num_samples = np.array(num_samples).reshape(-1, 1)
    epoch_stats = np.sum(
        pd.DataFrame(epoch_stats) * num_samples, axis=0
    ) / num_samples.sum()

    epoch_stats = epoch_stats.to_dict()
    epoch_stats.update({"train": "Train" if train else "Test"})
    return epoch_stats

def train(arglist):
    torch.autograd.set_detect_anomaly(arglist.debug)
    print(f"learning from demonstrations with: {arglist}\n")

    # load data
    file_path = os.path.join(arglist.data_path, "processed_data.csv")
    df = pd.read_csv(file_path)
    
    train_loader, test_loader = train_test_split(
        df,
        arglist.vi_batch_size,
        arglist.train_ratio,
        arglist.iterative,
        arglist.seed,
    )
    arglist.vi_batch_size = train_loader.batch_size

    start_time = time.time()
    history = []
    model = init_model(arglist)
    print(model)

    for e in range(arglist.epochs):
        # train 
        train_stats = run_epoch(model, train_loader, train=True)
        train_stats = {
            **{"epoch": e, "time": time.time() - start_time}, 
            **train_stats
        }
        history.append(train_stats)
        
        if (e + 1) % arglist.verbose == 0:
            print({
                key: np.round(val, 3) if not isinstance(val, str) else val 
                for key, val in train_stats.items()
            }, "\n")
        
        # test 
        if test_loader is not None:
            test_stats = run_epoch(model, test_loader, train=False)
            test_stats = {
                **{"epoch": e, "time": time.time() - start_time}, 
                **test_stats
            }
            history.append(test_stats)
            
            if (e + 1) % arglist.verbose == 0:
                print({
                    key: np.round(val, 3) if not isinstance(val, str) else val 
                    for key, val in test_stats.items()
                }, "\n")

    df_history = pd.DataFrame(history)
    
    # save results
    if arglist.save:
        exp_path = arglist.exp_path
        save_path = os.path.join(
            exp_path, 
            "state_dim_{}_prior_dist_{}_num_components_{}_post_dist_{}"
            "_iterative_{}_init_uniform_{}_obs_penalty_{}_prior_lr_{}"
            "_vi_lr_{}_seed_{}".format(
                arglist.state_dim, 
                arglist.prior_dist, 
                arglist.num_components, 
                arglist.post_dist, 
                arglist.iterative, 
                arglist.init_uniform, 
                arglist.obs_penalty, 
                arglist.prior_lr, 
                arglist.vi_lr, 
                arglist.seed
            )
        )
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save args
        with open(os.path.join(save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        # save model
        torch.save(model.state_dict(), os.path.join(save_path, f"model.pt"))

        # save history
        df_history.to_csv(os.path.join(save_path, "history.csv"), index=False)

        print("model saved")
    return model, df_history

if __name__ == "__main__":
    arglist = parse_args()
    model, df_history = train(arglist)