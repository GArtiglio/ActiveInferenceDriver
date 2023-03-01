import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

import torch

from src.ebirl import EBIRL
from src.data import pad_collate
from src.simulation import Simulator, simulate

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument("--data_path", type=str, default="../../data", help="data path, default=../../data")
    parser.add_argument("--exp_path", type=str, default="../../exp", help="experiment path, default=../exp")
    parser.add_argument("--exp_name", type=str, default="", help="saved experiment name")
    # result args
    parser.add_argument("--max_steps", type=int, default=130, help="max simulation steps, default=130")
    parser.add_argument("--save", type=bool_, default=True)
    parser.add_argument("--seed", type=int, default=0)
    arglist = parser.parse_args()
    arglist = vars(arglist)
    return arglist

def main(arglist):
    np.random.seed(arglist["seed"])
    torch.manual_seed(arglist["seed"])
    print(f"testing with args: {arglist}\n")
    
    # load data
    with open(os.path.join(arglist["data_path"], "data.p"), "rb") as f:
        data = pickle.load(f)

    # parse data
    drive_ids = [d["drive_id"] for d in data]
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
    obs_np = (nan_mask.unsqueeze(-1) * data["obs"]).numpy()
    
    obs_mean = np.nanmean(obs_np, axis=(0, 1))
    obs_std = np.nanstd(obs_np, axis=(0, 1))

    obs = (data["obs"] - obs_mean) / obs_std
    act = data["act"].long()

    obs_mask = obs * nan_mask.unsqueeze(-1)
    act_mask = act * nan_mask
    
    # load args
    exp_path = os.path.join(arglist["exp_path"], arglist["exp_name"])
    with open(os.path.join(exp_path, "args.json"), "rb") as f:
        config = json.load(f)
    
    # load state dict
    state_dict = torch.load(os.path.join(exp_path, "model.pt"), map_location="cpu")

    # init model
    batch_size = obs.shape[1]
    act_dim = len(torch.unique(act))
    obs_dim = obs.shape[-1]
    model = EBIRL(
        config["state_dim"], act_dim, obs_dim, config["horizon"], 
        prior_cov=config["prior_cov"], bc_penalty=config["bc_penalty"], 
        obs_penalty=config["obs_penalty"], kl_penalty=config["kl_penalty"]
    )
    model.init_q(batch_size, freeze_prior=False)
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    agent = model.agent

    print(agent)

    # get posterior parameters
    with torch.no_grad():
        q_dist = model.get_q_dist()
        q_mean = q_dist.mean

    # simulate agent
    results = []
    for eps_id in range(obs.shape[1]):
        obs_seq = obs_mask[:, eps_id, 0]
        act_seq = act_mask[:, eps_id]
        obs_min = (0. - obs_mean[0]) / obs_std[0]
        t_mean = obs_mean[1]
        t_std = obs_std[1]
        max_steps = arglist["max_steps"]
        env = Simulator(obs_seq, act_seq, obs_min, t_mean, t_std)
        obs_sim, act_sim, pi, b = simulate(
            env, agent, q_mean[eps_id], max_steps
        )
        
        rt_true = torch.where(act_seq == 0)[0][0].data.item()
        rt_pred = torch.where(act_sim == 0)[0]
        rt_pred = len(obs) if len(rt_pred) == 0 else rt_pred[0].data.item()
        
        results.append({
            "drive_id": drive_ids[eps_id],
            "rt_true": rt_true,
            "rt_pred": rt_pred
        })
    
    df_results = pd.DataFrame(results)

    if arglist["save"]:
        df_results.to_csv(os.path.join(exp_path, "simulation.csv"), index=False)

        print(f"simulation results saved at: {exp_path}")

if __name__ == "__main__":
    arglist = parse_args()
    main(arglist)