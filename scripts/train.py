import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd

import torch

from src.ebirl import EBIRL
from src.data import pad_collate
from src.train_utils import train, Logger

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../../data", help="data path, default=../../data")
    parser.add_argument("--exp_path", type=str, default="../exp", help="experiment path, default=../exp")
    # agent args
    parser.add_argument("--state_dim", type=int, default=2, help="agent state dimension, default=2")
    parser.add_argument("--horizon", type=int, default=30, help="agent max plan horizon, default=30")
    # prior args
    parser.add_argument("--prior_cov", type=str, choices=["diag", "full"], default="full", help="prior covariance type, default=full")
    parser.add_argument("--bc_penalty", type=float, default=1., help="prior behavior cloning penalty, default=1.")
    parser.add_argument("--obs_penalty", type=float, default=1., help="observation likelihood penalty, default=1.")
    parser.add_argument("--prior_penalty", type=float, default=1., help="prior kl penalty, default=1.")
    # train args
    parser.add_argument("--cp_path", type=str, default="none", help="checkpoint path, default=none")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cp_every", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--seed", type=int, default=0)
    arglist = parser.parse_args()
    arglist = vars(arglist)
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"learning with args: {arglist}\n")
    
    # load data
    with open(os.path.join(arglist["data_path"], "data.p"), "rb") as f:
        data = pickle.load(f)
    
    # parse data
    data = [
        {
            k: torch.from_numpy(v).to(torch.float32) 
            for k, v in d.items() if k not in ["drive_id", "t"]
        } for d in data
    ]
    data, mask = pad_collate(data)
    
    # process data
    nan_mask = mask.clone()
    nan_mask[mask == 0] = torch.nan
    obs_np = (nan_mask * data["obs"]).numpy()
    
    obs_mean = np.nanmean(obs_np)
    obs_std = np.nanstd(obs_np)

    obs = (data["obs"] - obs_mean) / obs_std
    act = data["act"].long()

    data = (obs, act, mask)
    
    # init model
    batch_size = obs.shape[1]
    act_dim = len(torch.unique(act))
    model = EBIRL(
        arglist["state_dim"], act_dim, arglist["horizon"], 
        prior_cov=arglist["prior_cov"], bc_penalty=arglist["bc_penalty"], 
        obs_penalty=arglist["obs_penalty"], prior_penalty=arglist["prior_penalty"]
    )
    model.init_q(batch_size, freeze_prior=False)
    print(model)
        
    # init loss function
    loss_fn = lambda m, o, a, mask: m.compute_loss(o, a, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=arglist["lr"], weight_decay=arglist["decay"])
    
    # load pretrained prior
    cp_history = None
    if arglist["cp_path"] != "none":
        cp_path = os.path.join(arglist["exp_path"], arglist["cp_path"])
        cp_model_path = glob.glob(os.path.join(cp_path, "models/*.pt"))
        cp_model_path.sort(key=lambda x: int(os.path.basename(x).replace(".pt", "").split("_")[-1]))
        
        state_dict = torch.load(cp_model_path[-1], map_location="cpu")
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        
        # load history
        cp_history = pd.read_csv(os.path.join(cp_path, "history.csv"))
        print(f"loaded checkpoint from {cp_path}")
    
    callback = None
    if arglist["save"]:
        callback = Logger(arglist, model.plot_keys, cp_history) 

    # train loop
    model, optimizer, df_history, callback = train(
        model, data, loss_fn, optimizer, arglist["epochs"], 
        verbose=arglist["verbose"], callback=callback, grad_check=False
    )

    if arglist["save"]:
        callback.save_checkpoint(model, optimizer)
        callback.save_history(df_history)
    return model, df_history

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)