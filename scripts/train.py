import argparse
import os
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.ebirl import EBIRL
from src.data import DriverDataset, pad_collate
from src.train_utils import train, Logger

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../dat/processed", help="data path, default=../dat/processed")
    parser.add_argument("--exp_path", type=str, default="../exp", help="experiment path, default=../exp")
    # agent args
    parser.add_argument("--state_dim", type=int, default=2, help="agent state dimension, default=2")
    parser.add_argument("--horizon", type=int, default=35, help="agent max plan horizon, default=35")
    parser.add_argument("--obs_dist", type=str, choices=["lognorm", "norm"])
    parser.add_argument("--a_cum", type=bool_, default=False, help="whether to use cumulative parameterization for obs mean, default=False")
    # prior args
    parser.add_argument("--prior_cov", type=str, choices=["diag", "full"], default="full", help="prior covariance type, default=full")
    parser.add_argument("--bc_penalty", type=float, default=1., help="prior behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=1., help="observation likelihood penalty, default=1.")
    parser.add_argument("--prior_penalty", type=float, default=1., help="prior kl penalty, default=1.")
    # train args
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--t_add", type=int, default=3, help="additional braking time steps for training, default=3")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cp_every", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--seed", type=int, default=0)
    arglist = parser.parse_args()
    return arglist

def main(arglist):
    np.random.seed(arglist.seed)
    torch.manual_seed(arglist.seed)
    print(f"learning with args: {arglist}\n")
    
    # load data
    file_path = os.path.join(arglist.data_path, "processed_data.csv")
    df = pd.read_csv(file_path)

    dataset = DriverDataset(df, t_add=arglist.t_add)
    batch_size = len(dataset)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=pad_collate
    )

    # init model
    act_dim = 2
    model = EBIRL(
        arglist.state_dim, act_dim, arglist.horizon, 
        obs_dist=arglist.obs_dist, a_cum=arglist.a_cum, 
        prior_cov=arglist.prior_cov, bc_penalty=arglist.bc_penalty, 
        obs_penalty=arglist.obs_penalty, prior_penalty=arglist.prior_penalty
    )
    model.init_q(batch_size, freeze_prior=False)
    print(model)
        
    # init loss function
    loss_fn = lambda m, o, a, mask: m.compute_loss(o, a, mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arglist.lr, weight_decay=arglist.decay)
    
    # load pretrained prior
    cp_history = None
    if arglist.cp_path != "none":
        cp_path = os.path.join(arglist.exp_path, arglist.cp_path)
        cp_model_path = glob.glob(os.path.join(cp_path, "models/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1], map_location="cpu")
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        
        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        print(f"loaded checkpoint from {cp_path}")
    
    callback = None
    if arglist.save:
        callback = Logger(arglist, model.plot_keys, cp_history) 

    # train loop
    model, optimizer, df_history, callback = train(
        model, loader, loss_fn, optimizer, arglist.epochs, 
        verbose=arglist.verbose, callback=callback, grad_check=False
    )

    if arglist.save:
        callback.save_checkpoint(model, optimizer)
        callback.save_history(df_history)
    return model, df_history

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)