import argparse
import os
import json
import pickle
import random
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import FactorAnalysis
from kneed import KneeLocator

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader

from preprocess import calc_looming
from train import DriverDataset
from src.legacy.ebirl import EBIRL

import warnings
warnings.filterwarnings("ignore")

# set pandas display option
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

def load_args(path):
    """ load training args """
    with open(path, "rb") as f:
        config = json.load(f)
    
    # bad exception handle
    if "num_components" not in config.keys():
        config.num_components = 3
    return config

def load_model(config, state_dict):
    model = EBIRL(
        config["state_dim"],
        config["obs_dim"],
        config["act_dim"],
        config["horizon"],
        config["precision"],        
        config["rnn_hidden_dim"],
        config["rnn_stack_dim"],
        config["rnn_dropout"],
        rnn_bidirectional=config["rnn_bidirectional"],
        # prior settings
        prior_dist=config["prior_dist"],
        num_components=config["num_components"],
        # post settings
        post_dist=config["post_dist"],
        num_flows=config["num_flows"],
        iterative=config["iterative"],
        batch_size=config["vi_batch_size"],
        init_uniform=config["init_uniform"],
        logvar_offset=config["logvar_offset"],
        # learning settings
        prior_lr=config["prior_lr"],
        prior_decay=config["prior_decay"],
        prior_schedule_steps=config["prior_schedule_steps"],
        prior_schedule_gamma=config["prior_schedule_gamma"],
        vi_lr=config["vi_lr"],
        vi_decay=config["vi_decay"],
        vi_schedule_steps=config["vi_schedule_steps"],
        vi_schedule_gamma=config["vi_schedule_gamma"],
        mc_samples=config["mc_samples"],
        obs_penalty=config["obs_penalty"],
        kl_penalty=config["kl_penalty"],
        kl_anneal_steps=config["kl_anneal_steps"],
        kl_const_steps=config["kl_const_steps"],
        kl_cycles=config["kl_cycles"],
        grad_clip=config["grad_clip"],
        seed=config["seed"]
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def sample_prior(model, num_samples):
    """
    Args:
        model (object): EBIRL model
        num_samples (int): number of samples

    Returns:
        params (torch.tensor): prior samples [num_samples, num_params]
        logp (torch.tensor): [num_samples]
    """
    params, logp = model.prior.sample(num_samples)    
    return params.squeeze(1), logp

def sample_posterior(df, model, num_samples):
    """
    Args:
        df (pd.dataframe): dataset
        model (object): EBIRL model
        num_samples (int): number of samples

    Returns:
        z (torch.tensor): posterior samples [num_samples, batch_size, num_params]
        logp (torch.tensor): [num_samples, batch_size]
    """
    dataset = DriverDataset(df)
    loader = DataLoader(dataset, batch_size=len(df["episode"].unique()), shuffle=False)
    for data_batch in loader:
        obs, act, mask = data_batch
        with torch.no_grad():
            # infer latent factor
            obs_mask = obs * mask.unsqueeze(-1)
            obs_mask[mask.unsqueeze(-1) == 0] = -1
            act_mask = act * mask
            act_mask[mask == 0] = model.act_dim
            act_mask = F.one_hot(act_mask, model.act_dim + 1)
            
            z_dist = model.encoder(torch.cat([obs_mask, act_mask], dim=-1))
            z, logp = model.encoder.sample(z_dist, num_samples)
    return z, logp

def compute_likelihood(df, model):
    batch_size = len(df["episode"].unique())

    dataset = DriverDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for data_batch in loader:
        obs, act, mask = data_batch
        with torch.no_grad():
            logp_a, logp_o, logp_post, logp_prior = model(obs, act, mask)
            kl = torch.mean(logp_post - logp_prior, dim=0)
            logp_a_mask = torch.sum(logp_a * mask.unsqueeze(1), dim=-1)
            logp_o_mask = torch.sum(logp_o * mask.unsqueeze(1), dim=-1) 
            elbo = logp_a_mask.mean(1) - kl
        
        # per step log probs
        logp_o_step = logp_o_mask / mask.sum(-1, keepdim=True)  
        
        # per step action prob
        p_step = torch.sum(logp_a.exp() * mask.unsqueeze(1), -1) / mask.sum(-1, keepdim=True)
        
        mask_ = mask.clone()
        mask_[mask == 0] = 10000
        p_min, _ = torch.min(logp_a.exp() * mask_.unsqueeze(1), dim=-1)
    
    stats_dict = {
        "elbo": elbo,
        "logp_a": logp_a_mask,
        "logp_o": logp_o_step,
        "kl": kl,
        "p_step": p_step,
        "p_min": p_min
    }
    return stats_dict

def transform_params(params, agent, sort=False, transform_lognorm=False):
    """ 
    Args:
        params (torch.tensor): [batch_size, num_params]
        agent (object): ActiveInference agent
        sort (bool, optional): Whether to sort A_mean. Defaults to False.
        transform_lognorm (bool, optional): Whether to tranform lognorm stats. Defaults to False.

    Returns:
        df_params (pd.dataframe): [batch_size, num_params]
    """
    (A_mean, A_std, B, C, D, inv_tau, inv_beta) = agent.transform_params(params)
    state_dim = B.shape[2]
    
    # sort params by A_mean
    batch_size = len(A_mean)
    if sort:
        idx = A_mean.argsort(dim=-1).view(batch_size, -1)
        A_mean = torch.gather(A_mean.squeeze(1), 1, idx)
        A_std = torch.gather(A_std.squeeze(1), 1, idx)
        B = torch.gather(B, 2, idx.view(batch_size, 1, -1, 1).repeat(1, 2, 1, 2))
        B = torch.gather(B, 3, idx.view(batch_size, 1, 1, -1).repeat(1, 2, 2, 1))
        C = torch.gather(C, 1, idx)
        D = torch.gather(D, 1, idx)
    
    params = [A_mean, A_std, B, C, D, inv_tau, inv_beta]
    params = [p.view(batch_size, -1) for p in params]
    params = torch.cat(params, dim=-1)

    # manual column header
    A_mean_labels = [r"$A_{\mu%s}$"%(i) for i in range(A_mean.shape[-1])]
    A_std_labels = [r"$A_{\sigma%s}$"%(i) for i in range(A_std.shape[-1])]
    B_labels = [
        r"$B^{%s}_{%s%s}$"%(i, j, k) for i in range(B.shape[1]) \
            for j in range(B.shape[2]) for k in range(B.shape[3])
    ]
    C_labels = [r"$C_{%s}$"%(i) for i in range(C.shape[1])]
    D_labels = [r"$D_{%s}$"%(i) for i in range(D.shape[1])]
    tau_labels = [r"$\tau$"]
    gamma_labels = [r"$\gamma$"]
    labels = A_mean_labels + A_std_labels + B_labels + C_labels + D_labels + \
        tau_labels + gamma_labels
    
    df_params = pd.DataFrame(params.numpy(), columns=labels)
    
    def get_lognorm_stats(loc, scale, return_mode=False):
        if not return_mode:
            mean = np.exp(loc + scale ** 2 / 2)
        else:
            mean = np.exp(loc - scale ** 2)
        variance = (np.exp(scale ** 2) - 1) * np.exp(2 * loc + scale ** 2)
        return mean, variance

    if transform_lognorm:
        for i in range(state_dim):
            obs_mean, obs_variance = get_lognorm_stats(
                df_params[r"$A_{\mu%s}$"%(i)], 
                df_params[r"$A_{\sigma%s}$"%(i)],
                return_mode=True
            )
            df_params[r"$A_{\mu%s}$"%(i)] = obs_mean
            df_params[r"$A_{\sigma%s}$"%(i)] = np.sqrt(obs_variance)
        
        # transform tau
        df_params[r"$\tau$"] = 1 / np.clip(df_params[r"$\tau$"], 1e-3, 1e5)
    
    return df_params

def pairwise_ks_test(samples, num_tests=100, seed=0):
    """ KS test for pairs of posterior distributions

    Args:
        samples (np.array): [num_samples, batch_size, num_vars]
        num_tests (int, optional): Number of sample tests to perform. Defaults to 100.
        seed (int, optional): Defaults to 0.

    Returns:
        ks (np.array): [num_tests, num_vars]
    """
    random.seed(seed)
    [_, batch_size, num_vars] = samples.shape
    test_pairs = list(itertools.combinations(np.arange(batch_size), 2))
    
    # subsample tests
    num_tests = min(num_tests, len(test_pairs))
    random.shuffle(test_pairs)
    test_pairs = test_pairs[:num_tests]

    ks = np.zeros((len(test_pairs), num_vars))
    for i in range(num_vars):
        for j in range(num_tests):
            sample1 = samples[:, test_pairs[j][0], i]
            sample2 = samples[:, test_pairs[j][1], i]
            ks[j, i], _ = stats.ks_2samp(sample1, sample2)
    return ks

def factor_selection(df, num_factors, rotation=None, num_seeds=1):
    scores = np.zeros((len(num_factors), num_seeds))
    bic = np.zeros((len(num_factors), num_seeds))
    for i in range(len(num_factors)):
        for j in range(num_seeds):
            model = FactorAnalysis(
                n_components=num_factors[i], 
                tol=1e-5,
                max_iter=10000, 
                svd_method="lapack", 
                rotation=rotation,
                random_state=j
            )
            model.fit(df)
            
            # collect stats
            scores[i,j] = model.score(df)
            bic[i,j] = (len(model.components_.flatten()) + \
                        len(model.noise_variance_) + \
                        len(model.mean_)) * np.log(df.shape[0]) - 2 * model.loglike_[-1]
                
    return {"num_factors": num_factors, "scores": scores, "BIC": bic}

def fit_predict_factor(df, num_factors, rotation=None, seed=0):
    model = FactorAnalysis(n_components=num_factors, 
        tol=1e-5, 
        max_iter=10000, 
        svd_method="lapack", 
        rotation=rotation,
        random_state=seed
    )
    model.fit(df)
    w = model.components_
    var = model.noise_variance_
    mu = model.mean_
    z = model.transform(df) 

    cols = df.columns
    df_w = pd.DataFrame(w, columns=cols, index=[i + 1 for i in range(num_factors)])
    df_var = pd.DataFrame(var.reshape(1, -1), columns=cols)
    df_mu = pd.DataFrame(mu.reshape(1, -1), columns=cols)
    df_z = pd.DataFrame(z, columns=[f"Factor {i+1}" for i in range(num_factors)])

    return {"w": df_w, "var": df_var, "mu": df_mu, "z": df_z}

def factor_analysis(df, normalize=True, rotation=None, seed=0):
    """ Perform factor analysis on model parameters

    Args:
        df (pd.dataframe): Data to use for factor analysis. 
            The dataframe should only contain columns of variables and no other labeling columns.
        normalize (bool, optional): Normalize input. Defaults to True.
        rotation ([str, None], optional): Apply factor rotation, "varimax" or None. 
            Defaults to None.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        dict_selection (dict): Factor selection result dict {"num_factors", "scores", "BIC"}
        dict_fa (dict): Factor analysis result dict 
            {"w", "var", "mu", "z", "norm_mu", "norm_std", "num_factors"}
    """
    # normalize
    if normalize:
        mu = df.mean()
        std = df.std()
        df = (df - mu) / (std + 1e-6)

    # factor selection
    num_factors = np.arange(1, 10 + 1)
    dict_selection = factor_selection(df, num_factors, rotation=rotation)
    criterion = dict_selection["BIC"].mean(-1)
    kneedle = KneeLocator(
        num_factors, criterion, 
        S=1, curve="convex", direction="decreasing"
    )

    # run optimal factors
    num_factor = kneedle.knee
    dict_fa = fit_predict_factor(df, num_factor, rotation, seed=seed)
    dict_fa.update({
        "norm_mu": mu if normalize else 0,
        "norm_std": std if normalize else 1,
        "num_factors": num_factor
    })
    return dict_selection, dict_fa

def sample_factor(dict_fa, num_samples, std, seed=0):
    """ Sample uniformly from factor model

    Args:
        dict_fa (dict): Factor analysis results
        num_samples (int): Number of samples
        std (float): Sampling standard deviation
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        z (torch.tensor): Latent factor samples [num_samples, num_params]
        df_factor (pd.dataframe): Observations generated from z [num_sample, num_factors]
    """
    np.random.seed(seed)
    num_factors = dict_fa["num_factors"]
    eps = np.random.uniform(-std, std, size=(num_samples, num_factors))
    z = eps.dot(dict_fa["w"].values) + dict_fa["mu"].values
    z = z * dict_fa["norm_std"].values + dict_fa["norm_mu"].values
    z = torch.from_numpy(z.astype(np.float32))
    df_factor = pd.DataFrame(eps, columns=[f"factor_{i}" for i in range(num_factors)])
    return z, df_factor

def compute_cf_looming(ev_profile, lv_profile, total_time, dt=1/10, plot=False):
    """ Compute counterfactual looming sequence with input vehicle profiles

    Args:
        ev_profile (dict): {
            "pos"; initial pos [batch_size], 
            "vel": initial vel [batch_size], 
            "acc": list of acc tuples [(t, acc)]
            }
        lv_profile (dict): same as ev_profile
        total_time (int): total time steps
        dt (float, optional): time increment. Defaults to 1/10.
        plot (bool, optional): whether to plot for debugging. Defaults to False.

    Returns:
        tau (np.array): looming sequence [batch_size, total_time]
    """
    # get acc sequence
    ev_acc = [ev_profile["acc"][1][ev_profile["acc"][0].index(t)] \
        if t in ev_profile["acc"][0] else np.nan for t in range(total_time)]
    ev_acc = pd.DataFrame(ev_acc).fillna(method="ffill").to_numpy().flatten()
    lv_acc = [lv_profile["acc"][1][lv_profile["acc"][0].index(t)] \
        if t in lv_profile["acc"][0] else np.nan for t in range(total_time)]
    lv_acc = pd.DataFrame(lv_acc).fillna(method="ffill").to_numpy().flatten()

    ev_vel = ev_profile["vel"].reshape(-1, 1) + np.cumsum(ev_acc * dt).reshape(1, -1)
    ev_pos = ev_profile["pos"].reshape(-1, 1) + np.cumsum(ev_vel * dt, axis=-1)
    lv_vel = lv_profile["vel"].reshape(-1, 1) + np.cumsum(lv_acc * dt).reshape(1, -1)
    lv_pos = lv_profile["pos"].reshape(-1, 1) + np.cumsum(lv_vel * dt, axis=-1)
    
    D = lv_pos - ev_pos
    v_rel = ev_vel - lv_vel
    tau = calc_looming(D, v_rel)
    # clip looming to be positive
    tau = np.clip(tau, 1e-6, 100)
    
    if plot:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(ev_pos[0], label="ev_pos")
        ax[0].plot(ev_vel[0], label="ev_vel")
        ax[0].plot(lv_pos[0], label="lv_pos")
        ax[0].plot(lv_vel[0], label="lv_vel")
        ax[0].legend()
        ax[1].plot(tau[0], "-o", label="tau")
        plt.show()
    return tau

def compute_cf_rt(
    agent, params, ev_profile, lv_profile, T, method="monte_carlo", num_samples=50, seed=0
    ):
    """ Compute counterfactual reaction times

    Args:
        agent (object): ActiveInference agent
        params (torch.tensor): ActiveInference agent parameters
        ev_profile (dict): Ego vehicle trajectory profile
        lv_profile (dict): Lead vehicle trajectory profile
        T (int): Max simulation time
        method (str, optional): Reaction time computation method. 
            "geometric" computes analytical rts but could be inaccurate with low entropy actions. 
            "monte_carlo" takes sample average. Defaults to "monte_carlo".
        num_samples (int, optional): Number of samples for "monte_carlo" method. Defaults to 50.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        rt (torch.tensor): Simulated reaction times [batch_size, 1]
        logp_o (torch.tensor): Observation likelihood of counterfactual looming [batch_size, 1]
    """
    torch.manual_seed(seed)
    batch_size = len(params)
    tau = compute_cf_looming(ev_profile, lv_profile, T)
    
    # inference
    with torch.no_grad():
        obs = torch.from_numpy(tau).unsqueeze(-1)
        act = torch.ones(batch_size, T).long()
        logp_a, logp_o = agent(params, obs, act)
    
    p_a = logp_a.exp()

    if method == "geometric":
        p = torch.cumprod(1 - p_a, dim=-1) * p_a
        rt = torch.argmax(p[:, 1:], dim=1).numpy() + 1
    elif method == "monte_carlo":
        p_a = torch.stack([1 - p_a, p_a]).clip(0, 1).permute(1, 2, 0)
        act = dist.Categorical(probs=p_a).sample((num_samples,))
        act_cum = torch.cumsum(act, dim=-1)
        # fill in never brake
        act_cum[:, :, -1] += 1 
        
        # find first brake
        idx_rt = torch.argmin(torch.abs(act_cum - 1), dim=-1).float()
        rt = torch.mean(idx_rt, dim=0).numpy()
    
    # compute logp_o
    mask = torch.zeros(batch_size, T)
    mask[torch.arange(batch_size), rt.round()] = 1
    mask = 1 - torch.cumsum(mask, dim=-1)
    logp_o = torch.sum(logp_o * mask, dim=-1) / mask.sum(-1)
    
    # find looming at rt
    tau_rt = tau[np.arange(len(rt)).astype(int), np.round(rt).astype(int)]
    return rt, logp_o.numpy(), tau_rt

def sample_predictive_check(agent, params, df_data, lv_acc, fit=True, T=30, seed=0):
    """ Prior and posterior predictive check from samples

    Args:
        agent (object): ActiveInference agent object
        params (torch.tensor): Active inference params 
        df_data (pd.dataframe): Driving sim data to fit initial conditions with fields 
            ["speed_ms", "d"] 
        lv_acc (float): Lead vehicle acceleration
        fit (bool, optional): Whether to fit initial condition. 
            If not fit sample initial condition from data. Defaults to True.
        T (int, optional): Max simulation time. Defaults to 30.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        rt (torch.tensor): Simulated reaction times [batch_size, 1]. 
        logp_o (torch.tensor): Observation likelihood of counterfactual looming [batch_size, 1].
    """
    np.random.seed(seed)
    df_data = df_data.groupby("episode").head(1).reset_index(drop=True)
    num_samples = len(params)
    
    if fit:
        # fit init speed
        params_speed = stats.t.fit(df_data["speed_ms"])
        speed_samples = stats.t.rvs(
            loc=params_speed[-2], 
            scale=params_speed[-1], 
            size=num_samples, 
            random_state=seed, 
            *params_speed[:-2]
        )

        # fit init dist
        params_d = stats.halfnorm.fit(df_data["d"])
        d_samples = stats.halfnorm.rvs(
            loc=params_d[-2], 
            scale=params_d[-1], 
            size=num_samples, 
            random_state=seed, 
            *params_d[:-2]
        )
        
    else:
        idx = np.random.randint(0, high=len(df_data), size=(num_samples))
        d_samples = df_data["d"][idx].values
        speed_samples = df_data["speed_ms"][idx].values
    
    lv_profile = {"pos": d_samples, "vel": 28 * np.ones((num_samples,)), "acc":[(0,), (lv_acc,)]}
    ev_profile = {"pos": np.zeros((num_samples,)), "vel": speed_samples, "acc":[(0,), (0,)]}
    rt, logp_o, tau_rt = compute_cf_rt(agent, params, ev_profile, lv_profile, T, seed=seed)
    return rt, logp_o

def counterfactual_simulations(dict_fa, agent, ev_delays, lv_acc, num_samples, std, T=30, seed=0):
    """ Counterfactual simulations from factor model

    Args:
        dict_fa (dict): Factor analysis results
        agent (object): ActiveInference agent result
        ev_delays (list): Ego vehicle delay time step values
        lv_acc (float): Lead vehicle acceleration values
        num_samples (int): Number of samples
        std (float): Sampling std
        T (int, optional): Max simulation time. Defaults to 30.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        df_cf (pd.dataframe): Simulation result dataframe with fields ["Factor", "Delay", "RT"]
    """
    np.random.seed(seed)
    
    # sample from factor model
    num_factors = dict_fa["num_factors"]
    cols = list(dict_fa["w"].columns)
    eps = np.random.uniform(-std, std, size=(num_samples, num_factors))
    z = eps.dot(dict_fa["w"].values) + dict_fa["mu"].values
    z = z * dict_fa["norm_std"].values + dict_fa["norm_mu"].values
    z = torch.from_numpy(z.astype(np.float32))
    
    # simulate delays
    lv_profile = {
        "pos":30 * np.ones(num_samples), 
        "vel":28 * np.ones(num_samples), 
        "acc":[(0,), (lv_acc,)]
    }
    df_cf = []
    for i, delay in enumerate(ev_delays):
        ev_profile = {
            "pos":0 * np.ones(num_samples), 
            "vel":28 * np.ones(num_samples), 
            "acc":[(0,), (-5,)] if delay == 0 else [(0, delay), (0, -5,)]
        }
        rt, _, tau_rt = compute_cf_rt(agent, z, ev_profile, lv_profile, T, seed=seed)
        
        df_delay = pd.DataFrame(
            np.hstack([eps, rt.reshape(-1, 1), tau_rt.reshape(-1, 1)]),
            columns=[f"Factor {i+1}" for i in range(num_factors)] + ["RT", "tau"]
        )
        df_delay["Delay"] = delay
        df_cf.append(df_delay)
        
    df_cf = pd.concat(df_cf, axis=0).reset_index(drop=True)
    return df_cf
    
def main(exp_path, data_path, sim_time=30, cf_lv_acc=-2, sort_obs=False, save=False, seed=0):
    torch.manual_seed(seed)
    # load args
    arglist = load_args(os.path.join(exp_path, "args.json"))

    # load model
    state_dict = torch.load(os.path.join(exp_path, "model_0.pt"))
    model = load_model(arglist, state_dict)
    
    # load data
    df_data = pd.read_csv(os.path.join(data_path, "processed_data.csv"))
    df_rt = df_data.groupby("episode").tail(1).reset_index(drop=True)
    df_rt.columns = [s.capitalize() for s in df_rt.columns]
    df_rt = df_rt.replace({"Scenario": {"HB": "C", "SB": "N"}})
    
    # load demographics 
    df_demo = pd.read_csv(os.path.join(data_path, "df_questionnaire.csv"), index_col=0)
    df_demo.columns = [s.capitalize() for s in df_demo.columns]
    
    # model performance stats
    stats_dict = compute_likelihood(df_data, model)
    
    """
    prepare results
    """
    print("prepare results...")
    # sample prior params
    num_samples = 1000
    prior_samples, logp_prior = sample_prior(model, num_samples)
    df_prior_transform = transform_params(
        prior_samples.data, model.agent, sort_obs, transform_lognorm=True
    )

    # sample posterior params
    num_samples = 1500
    post_samples, logp_post = sample_posterior(df_data, model, num_samples)
    post_map = post_samples[logp_post.argmax(0), torch.arange(logp_post.shape[1])]

    # split posterior into high and low
    idx_high = np.where(df_rt["Scenario"] == "C")[0]
    idx_low = np.where(df_rt["Scenario"] == "N")[0]
    post_samples_high = post_samples[:, idx_high]
    post_samples_low = post_samples[:, idx_low]

    df_post_transform = transform_params(
        post_samples.flatten(end_dim=-2), model.agent, sort_obs, transform_lognorm=True
    )
    df_post_map_transform = transform_params(
        post_map, model.agent, sort_obs, transform_lognorm=True
    )
    # convert time step to seconds
    df_post_transform[r"$\tau$"] /= 10
    df_post_map_transform[r"$\tau$"] /= 10

    df_post = transform_params(
        post_samples.flatten(end_dim=-2), model.agent, sort_obs, transform_lognorm=False
    )
    df_post_map = transform_params(
        post_map, model.agent, sort_obs, transform_lognorm=False
    )

    # post one drive params dist
    df_post_one_drive_transform = transform_params(
        post_samples[:, 0], model.agent, sort=sort_obs, transform_lognorm=True
    )

    # select effective params 
    cols = list(df_prior_transform.columns)
    A_cols = [r"$A_{\mu0}$", r"$A_{\mu1}$", r"$A_{\sigma0}$", r"$A_{\sigma1}$"]
    B_cols = [r"$B^{0}_{00}$", r"$B^{0}_{11}$", r"$B^{1}_{00}$", r"$B^{1}_{11}$"]
    C_cols = [r"$C_{0}$"]
    D_cols = [r"$D_{0}$"]
    tau_cols = [r"$\tau$"]
    gamma_cols = [r"$\gamma$"]
    eff_cols = A_cols + B_cols + C_cols + D_cols + tau_cols + gamma_cols

    df_prior_transform_eff = df_prior_transform[eff_cols]
    df_post_transform_eff = df_post_transform[eff_cols]
    df_post_map_transform_eff = df_post_map_transform[eff_cols]
    df_post_one_drive_transform_eff=  df_post_one_drive_transform[eff_cols] 

    # concat post with scenario
    df_post_transform_eff["Scenario"] = (
        df_rt["Scenario"] + "-" + df_rt["Alert"]
    ).values.reshape(1, -1).repeat(num_samples, 0).flatten()
    df_post_transform_eff["Scenario"] = pd.Categorical(
        df_post_transform_eff["Scenario"], 
        ordered=True, 
        categories=["C-A", "C-S", "N-A", "N-S"]
    )
    df_post_transform_eff["Participant_id"] = (
        df_rt["Participant_id"]
    ).values.reshape(1, -1).repeat(num_samples, 0).flatten()
    
    # concat post map with demographics 
    df_post_map_transform_eff_profile = pd.concat([df_post_map_transform_eff, df_rt], axis=1)
    df_post_map_transform_eff_profile = df_post_map_transform_eff_profile.merge(
        df_demo, on="Participant_id"
    )
    df_post_map_transform_eff_profile = df_post_map_transform_eff_profile.loc[
        df_post_map_transform_eff_profile["Gender"] != "Prefer not to answer"
    ]
    
    
    """
    prior and posterior predictive
    """
    print("prior and posterior predictive...")
    num_samples = 300
    lv_acc_low, lv_acc_high = -2, -5
    prior_samples_sub = prior_samples[torch.randint(len(prior_samples), (num_samples, ))]
    
    # split post into high and low
    post_samples_low_sub = post_samples_low.flatten(end_dim=-2)
    post_samples_low_sub = post_samples_low_sub[torch.randint(len(post_samples_low_sub), (num_samples,))]
    post_samples_high_sub = post_samples_high.flatten(end_dim=-2)
    post_samples_high_sub = post_samples_high_sub[torch.randint(len(post_samples_high_sub), (num_samples,))]
    
    T = 30
    rt_prior_low, logp_prior_low = sample_predictive_check(
        model.agent, prior_samples_sub, df_data, lv_acc_low, fit=True, T=T, seed=seed
    )
    rt_prior_high, logp_prior_high = sample_predictive_check(
        model.agent, prior_samples_sub, df_data, lv_acc_high, fit=True, T=T, seed=seed
    )

    rt_post_low, logp_post_low = sample_predictive_check(
        model.agent, post_samples_low_sub, df_data, lv_acc_low, fit=True, T=T, seed=seed
    )
    rt_post_high, logp_post_high = sample_predictive_check(
        model.agent, post_samples_high_sub, df_data, lv_acc_high, fit=True, T=T, seed=seed
    )
    rt_data_low = df_rt.loc[df_rt["Scenario"] == "N"]["Rt"]
    rt_data_high = df_rt.loc[df_rt["Scenario"] == "C"]["Rt"]
    
    # make tables
    df_predictive_low = pd.DataFrame.from_dict(
        {
            "rt": np.hstack([rt_prior_low, rt_post_low, rt_data_low]),
            "dist": ["prior"] * len(rt_prior_low) + \
                ["post"] * len(rt_post_low) + ["data"]  * len(rt_data_low)
        }
    )
    df_predictive_low["Scenario"] = "NC"
    
    df_predictive_high = pd.DataFrame.from_dict(
        {
            "rt": np.hstack([rt_prior_high, rt_post_high, rt_data_high]),
            "dist": ["prior"] * len(rt_prior_high) + \
                ["post"] * len(rt_post_high) + ["data"]  * len(rt_data_high)
        }
    )
    df_predictive_high["Scenario"] = "C"
    
    df_predictive = pd.concat([df_predictive_low, df_predictive_high], axis=0)
    
    """
    factor analysis
    """
    print("factor analysis...")
    # log transform factor input
    # factor analysis on post dist can be done but find assignment a bit hard
    # df_post_log = df_post.sample(n=5000, random_state=seed)
    # df_post_log.iloc[:, 2:] = np.log(df_post_log.iloc[:, 2:] + 1e-6)
    
    df_post_map_log = df_post_map
    df_post_map_log.iloc[:, 2:] = np.log(df_post_map_log.iloc[:, 2:] + 1e-6)
    dict_selection, dict_fa = factor_analysis(
        df_post_map_log, normalize=True, rotation="varimax", seed=seed
    )
    
    # select effective params for plotting
    df_factor_w_eff = dict_fa["w"][eff_cols]
    df_factor_var_eff = dict_fa["var"][eff_cols]

    # make tables
    df_factor_loading = pd.concat([df_factor_w_eff, df_factor_var_eff], axis=0)
    df_factor_loading["label"] = [f"factor_{i+1}" for i in range(len(df_factor_w_eff))] + ["var"]
    
    df_factor_assignment = pd.concat([df_rt, dict_fa["z"]], axis=1)
    df_factor_assignment["Scenario"] = df_factor_assignment["Scenario"] + "-" + df_factor_assignment["Alert"]
    df_factor_assignment["Scenario"] = pd.Categorical(
        df_factor_assignment["Scenario"], ordered=True, categories=["C-A", "C-S", "N-A", "N-S"]
    ) 
        
    """ 
    counterfactual simulations 
    """
    print("counterfactual simulation...")
    ev_delays = np.arange(0, sim_time, step=5)
    num_samples = 3000
    std = 5
    df_cf = counterfactual_simulations(
        dict_fa, model.agent, ev_delays, cf_lv_acc, num_samples, std, T=sim_time, seed=0
    )
    df_cf["Delay"] /= 10
    df_cf["RT"] /= 10
        
    """
    diagnostics
    """
    print("diagnostics...")
    # posterior ks test
    pairwise_ks_stats = pairwise_ks_test(post_samples.numpy(), num_tests=500, seed=0)
    df_pairwise_ks_stats = pd.DataFrame(pairwise_ks_stats, columns=cols)
    df_pairwise_ks_stats_eff = df_pairwise_ks_stats[eff_cols]
    
    # correlation analysis
    corr = np.corrcoef(np.log(df_post_map_transform_eff + 1e-6), df_rt["Rt"], rowvar=False)
    df_corr = pd.DataFrame(corr, columns=eff_cols + ["BRT"], index=eff_cols + ["BRT"])
    
    # interpolate and sample latent factor space 
    ev_delays = np.array([30])
    num_samples = 3000
    std = 5
    factor_samples, df_factor = sample_factor(
        dict_fa, num_samples, std, seed=0
    )
    df_factor_samples = transform_params(
        factor_samples, model.agent, sort=False, transform_lognorm=True
    )
    df_interpo = counterfactual_simulations(
        dict_fa, model.agent, ev_delays, cf_lv_acc, num_samples, std, T=sim_time, seed=0
    )
    df_interpo["RT"] /= 10
    df_factor_samples = pd.concat([df_factor_samples, df_interpo], axis=1)
    
    if save:
        print("saving results...")
        df_path = os.path.join(exp_path, "tab")
        if not os.path.exists(df_path):
            os.mkdir(df_path)
        
        with open(os.path.join(df_path, "learning_stats.p"), "wb") as f:
            pickle.dump(stats_dict, f)
            
        with open(os.path.join(df_path, "factor_selection.p"), "wb") as f:
            pickle.dump(dict_selection, f)
        
        df_prior_transform_eff.to_csv(os.path.join(df_path, "prior.csv"), index=False)
        df_post_transform_eff.to_csv(os.path.join(df_path, "post.csv"), index=False)
        df_post_map_transform_eff_profile.to_csv(os.path.join(df_path, "post_map.csv"), index=False)
        df_predictive.to_csv(os.path.join(df_path, "sample_predictive.csv"), index=False)
        df_pairwise_ks_stats_eff.to_csv(os.path.join(df_path, "pairwise_ks.csv"), index=False)
        df_corr.to_csv(os.path.join(df_path, "params_corr.csv"), index=True)
        
        df_factor_loading.to_csv(os.path.join(df_path, "factor_loading.csv"), index=False)
        df_factor_assignment.to_csv(os.path.join(df_path, "factor_assignment.csv"), index=False)
        df_factor_samples.to_csv(os.path.join(df_path, "factor_samples.csv"), index=False)
        df_cf.to_csv(os.path.join(df_path, "cf_simulation.csv"), index=False)
        
    print(f"results saved={save}")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", default="../dat/processed", help="data path, default=../dat/processed"
    )
    parser.add_argument(
        "--exp_path", default="../exp/ebirl", help="experiment path, default=../exp/ebirl"
    )
    parser.add_argument(
        "--exp_num", 
        default="lognormal_state_dim_2_embedding_dim_None_reward_ahead_"
            "True_prior_dist_mvn_num_components_3_post_dist_diag_iterative_True_init_"
            "uniform_True_obs_penalty_0.2_prior_lr_0.01_vi_lr_0.01_decay_0_seed_0", 
        help="experiment number, default=seed_0"
    )
    parser.add_argument(
        "--sim_time", type=int, default=50, 
        help="max counterfactual simulation time, default=50"
    )
    parser.add_argument(
        "--cf_lv_acc", type=float, default=-2, 
        help="counterfactual simulation lead vehicle acceleration, default=-2"
    )
    parser.add_argument("--save", type=bool_, default=True, help="save results, default=False")
    arglist = parser.parse_args()
    return arglist

if __name__ == "__main__":
    arglist = parse_args()
    result_path = os.path.join(arglist.exp_path, arglist.exp_num)
    
    main(
        result_path, 
        arglist.data_path, 
        sim_time=arglist.sim_time,
        cf_lv_acc=arglist.cf_lv_acc,
        save=arglist.save
    )