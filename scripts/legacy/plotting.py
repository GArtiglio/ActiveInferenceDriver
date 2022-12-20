import argparse
import os
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# set plotting style
strip_size = 12
label_size = 14
font_family = "times new roman"
mpl.rcParams["font.family"] = font_family
mpl.rcParams["axes.labelsize"] = label_size
mpl.rcParams["xtick.labelsize"] = strip_size
mpl.rcParams["ytick.labelsize"] = strip_size
mpl.rcParams["legend.title_fontsize"] = strip_size

def plot_learning_curve(df_history, show=False):
    """ assume no cv """
    df_train = df_history.loc[df_history["train"] == "Train"]
    
    fig, ax = plt.subplots(2, 4, figsize=(8, 4))
    ax = ax.flat
    ax[0].plot(df_train["epoch"], df_train["loss"], label="train_loss")
    ax[1].plot(df_train["epoch"], df_train["elbo"], label="train_elbo")
    ax[2].plot(df_train["epoch"], df_train["logp_a"], label="train_logp_a")
    ax[3].plot(df_train["epoch"], df_train["logp_o"], label="train_logp_o")
    ax[4].plot(df_train["epoch"], df_train["kl"], label="train_kl")
    ax[5].plot(df_train["epoch"], df_train["beta"], label="train_beta")
    ax[6].plot(df_train["epoch"], df_train["p_step"], label="train_p_step")
    ax[7].plot(df_train["epoch"], df_train["p_min"], label="train_p_min")

    if "Test" in df_history["train"].unique():
        df_test = df_history.loc[df_history["train"] == "Test"]
        ax[0].plot(df_test["epoch"], df_test["loss"], label="test_loss")
        ax[1].plot(df_test["epoch"], df_test["elbo"], label="test_elbo")
        ax[2].plot(df_test["epoch"], df_test["logp_a"], label="test_logp_a")
        ax[3].plot(df_test["epoch"], df_test["logp_a"], label="test_logp_o")
        ax[4].plot(df_test["epoch"], df_test["kl"], label="test_kl")
        ax[5].plot(df_test["epoch"], df_test["beta"], label="test_beta")
        ax[6].plot(df_test["epoch"], df_test["p_step"], label="test_p_step")
        ax[7].plot(df_test["epoch"], df_test["p_min"], label="test_p_min")
    
    for i in range(len(ax)):
        ax[i].legend()

    plt.tight_layout()

    if show:
        plt.show()
    return fig

def plot_stats_dist(stats_dict, show=False):
    """ plot learning performance stats """
    fig, ax = plt.subplots(2, 3, figsize=(6, 6))
    ax = ax.flat
    sns.histplot(stats_dict["elbo"], bins="sqrt", kde=True, ax=ax[0])
    ax[0].set_xlabel("ELBO"); 
    sns.histplot(stats_dict["logp_a"].flatten(), bins="sqrt", kde=True, ax=ax[1])
    ax[1].set_xlabel("logp_a")
    sns.histplot(stats_dict["logp_o"].flatten(), bins="sqrt", kde=True, ax=ax[2])
    ax[2].set_xlabel("logp_o")
    sns.histplot(stats_dict["kl"], bins="sqrt", kde=True, ax=ax[3])
    ax[3].set_xlabel("kl")
    sns.histplot(stats_dict["p_step"].flatten(), bins="sqrt", kde=True, ax=ax[4])
    ax[4].set_xlabel("p_step")
    sns.histplot(stats_dict["p_min"].flatten(), bins="sqrt", kde=True, ax=ax[5])
    ax[5].set_xlabel("p_min")
    
    for i in range(5):
        ax[i].set_ylabel("")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_params_hist(
    df_params, df_rt=None, scenario=None, figsize=6, sharey=False, lib="sns", show=False
    ):
    var_names = list(df_params.columns)
    num_vars = df_params.shape[1]
    num_rows = np.ceil(df_params.shape[1] / 3).astype(int)
    fig, ax = plt.subplots(num_rows, 3, figsize=(figsize, 1 * figsize), sharey=sharey)
    
    counter = 0
    for i in range(num_rows):
        for j in range(3):
            df_sub = df_params.iloc[:, counter].to_frame()
            if scenario is not None:
                df_sub["Scenario"] = scenario
            
            if lib == "sns":
                sns.histplot(
                    data=df_sub, 
                    x=var_names[counter], 
                    hue="Scenario" if scenario is not None else None,
                    bins="auto", stat="density", 
                    color="silver", legend=False, ax=ax[i, j]
                )
                sns.kdeplot(
                    data=df_sub, 
                    x=var_names[counter], 
                    hue="Scenario" if scenario is not None else None,
                    color="k", legend=False, ax=ax[i, j]
                )
            else:
                ax[i, j].hist(df_sub, density=True)
                ax[i, j].set_xlabel(list(df_params)[counter])
                
            # add rt on top of hist
            if df_rt is not None:
                ax[i, j].plot(df_sub.iloc[:, 0], df_rt["rt"], "o")

            # remove y labels
            if j > 0:
                ax[i, j].set_ylabel("")

            counter += 1
            if counter > num_vars:
                break

    plt.tight_layout(pad=0.2)

    if show:
        plt.show()
    return fig

def plot_params_scatter(df, show=False):
    """ Parameter scatter matrix """
    from matplotlib.ticker import FormatStrFormatter

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax = pd.plotting.scatter_matrix(df, ax=ax)
    for x in ax.flatten():
        x.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) 
    
    plt.tight_layout(w_pad=0.05, h_pad=0.05)
    if show:
        plt.show()
    return fig

def plot_params_corr(df_corr, show=False):
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(df_corr.round(2), annot=True, cbar=False, cmap="vlag", ax=ax)
    if show:
        plt.show()
    return fig

def plot_params_by_group(
    df_params, param_vars, by_var, plot_type="violin", 
    swarm=False, xtick_rot=0, num_cols=3, figsize=6, show=False
    ):
    """ boxplot """
    aspect = 0.8
    if plot_type == "violin":
        aspect = 0.5
    num_rows = np.ceil(len(param_vars) / num_cols).astype(int)
    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(figsize, aspect * figsize / num_cols * num_rows), sharex=True
    )
    ax = ax.reshape(num_rows, num_cols)

    counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if plot_type == "box":
                sns.boxplot(
                    data=df_params[[param_vars[counter], "Scenario"]], 
                    x="Scenario", y=param_vars[counter], ax=ax[i, j]
                )
            elif plot_type == "violin":
                sns.violinplot(
                    data=df_params[[param_vars[counter], by_var]], 
                    x=by_var, y=param_vars[counter], color="silver",
                    inner=None, ax=ax[i, j]
                )
            elif plot_type == "scatter":
                sns.scatterplot(
                    data=df_params[[param_vars[counter], by_var]],
                    x=by_var, y=param_vars[counter], color="k", ax=ax[i, j]
                )
            
            if swarm:
                sns.swarmplot(
                    data=df_params[[param_vars[counter], by_var]], 
                    x=by_var, y=param_vars[counter], 
                    color="k", size=4, ax=ax[i, j]
                )

            # custom axis
            # ax[i, j].set_ylabel(ax[i, j].get_ylabel(), rotation=0)
            if xtick_rot != 0:
                ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(), rotation=xtick_rot)
            
            # remove x label
            if i < num_rows - 1:
                ax[i, j].xaxis.set_visible(False)

            counter += 1
            if counter > len(param_vars):
                break
        
    plt.tight_layout(pad=0.4)
    if show:
        plt.show()
    return fig

def plot_factor_selection(dict, figsize=6, show=False):
    fig, ax = plt.subplots(2, 1, figsize=(figsize, 0.5 * figsize))
    ax[0].plot(dict["num_factors"], dict["scores"], "k-o")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel("Log-likelihood")

    ax[1].plot(dict["num_factors"], dict["BIC"], "k-o")
    ax[1].set_ylabel("BIC")
    ax[1].set_xlabel("Factors")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_factor_loadings(df_loading, figsize=4, vertical=False, show=True):
    df_w = df_loading.loc[df_loading["label"] != "var"].iloc[:, :-1]
    df_var = df_loading.loc[df_loading["label"] == "var"].iloc[:, :-1]
    df_w.index = np.arange(df_w.shape[0]) + 1
    
    df_var = 1. - df_var
    df_w = df_w.T.round(2)
    df_var = df_var.T.round(2)

    num_feat, num_factors = df_w.shape

    if vertical:
        nrows, ncols = 1, 2
        figsize = (figsize, 0.6 * num_feat * figsize / (num_factors + 1))
        h_ratios, w_ratios = [1], [df_w.shape[1], 1]
    else:
        df_w = df_w.T
        df_var = df_var.T
        nrows, ncols = 2, 1
        figsize = (0.75 * num_feat * figsize / (num_factors + 1), figsize)
        h_ratios, w_ratios = [0.4 * df_w.shape[1], 1], [1]
    
    annot_size = 12

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, \
        gridspec_kw=dict(height_ratios=h_ratios, width_ratios=w_ratios))
    sns.heatmap(data=df_w, annot=True, cmap="vlag", cbar=False, 
        annot_kws={"fontsize":annot_size, "family":font_family}, ax=ax[0])
    sns.heatmap(data=df_var, annot=True, cmap="Reds", cbar=False, 
        annot_kws={"fontsize":annot_size, "family":font_family}, ax=ax[1])
    
    if vertical:
        ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
        ax[1].yaxis.set_visible(False)
        ax[1].set_xticklabels([""])

        ax[0].set_ylabel("Parameter")
        ax[0].set_xlabel("Factor")
        ax[1].set_xlabel("Var. \nexplained")
    else:
        ax[0].xaxis.set_visible(False)
        ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
        ax[1].set_yticklabels([""])
        
        ax[0].set_ylabel("Factor")
        ax[1].set_xlabel("Parameter")
        ax[1].set_ylabel("Var. \nexplained")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_factor_vs_rt(df_factor_assignment, figsize=6, show=False):
    factor_cols = [c for c in df_factor_assignment.columns if "Factor" in c]
    num_factors = len(factor_cols)
    df_z = df_factor_assignment[factor_cols]

    fig, ax = plt.subplots(2, num_factors, figsize=(figsize, 0.6 * figsize), sharex="col")
    ax = ax.reshape(2, num_factors)
    for i in range(num_factors):
        # plot factor hist
        sns.histplot(
            df_z.iloc[:, i], 
            bins="sqrt", stat="density", 
            color="silver", ax=ax[0, i]
        )
        sns.kdeplot(df_z.iloc[:, i], color="k", ax=ax[0, i])
        
        # plot rt
        ax[1, i].plot(df_z.iloc[:, i], df_factor_assignment["Rt"]/10, "ko", markersize=4)
        
        # custom axis
        ax[0, i].xaxis.set_visible(False)
        ax[0, i].set_ylabel(ax[0, i].get_ylabel())
        
        ax[1, i].set_xlabel(f"Factor {i+1}")
        ax[1, i].set_ylabel("BRT (s)")

        if i > 0:
            ax[0, i].yaxis.set_visible(False)
            ax[1, i].yaxis.set_visible(False)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_sample_predictive(rt_prior, rt_post, rt_data, figsize=7, cumulative=True, show=False):
    # ks test
    ks_prior, p_prior = stats.ks_2samp(rt_prior, rt_data)
    ks_post, p_post = stats.ks_2samp(rt_post, rt_data)

    hist_type = "step"
    stat = "density"
    binrange = [0, np.hstack([rt_prior, rt_post, rt_data]).max()]

    fig, ax = plt.subplots(1, 2, figsize=(figsize, 0.35 * figsize), sharex=True, sharey=True)
    ax = ax.flat
    
    # plot prior
    sns.histplot(
        data=rt_prior, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=True, 
        stat=stat, label=f"Prior (KS={ks_prior:.3f})", ax=ax[0]
    )
    sns.histplot(
        data=rt_data, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=False, 
        stat=stat, label="Empirical", ax=ax[0]
    )

    ax[0].legend()
    ax[0].set_xlabel("BRT (s)")
    ax[0].set_ylabel("Cumulative density")

    # plot posterior
    sns.histplot(
        data=rt_post, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=True, 
        stat=stat, label=f"Posterior (KS={ks_post:.3f})", ax=ax[1]
    )
    sns.histplot(
        data=rt_data, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=False, 
        stat=stat, label="Empirical", ax=ax[1]
    )

    ax[1].legend()
    ax[1].set_xlabel("BRT (s)")
    ax[1].yaxis.set_visible(False)
    
    # custom axis
    ax[0].set_xticklabels(0.1 * ax[0].get_xticks().round(3))
    ax[1].set_xticklabels(0.1 * ax[1].get_xticks().round(3))

    plt.tight_layout()

    if show:
        plt.show()
    return fig

def plot_predictive(df_predictive, figsize=7, cumulative=True, show=False):
    rt_prior = df_predictive.loc[df_predictive["dist"] == "prior"]["rt"]
    rt_post = df_predictive.loc[df_predictive["dist"] == "post"]["rt"]
    rt_data = df_predictive.loc[df_predictive["dist"] == "data"]["rt"]
    
    # ks test
    ks_prior, p_prior = stats.ks_2samp(rt_prior, rt_data)
    ks_post, p_post = stats.ks_2samp(rt_post, rt_data)

    hist_type = "step"
    stat = "density"
    binrange = [0, np.hstack([rt_prior, rt_post, rt_data]).max()]

    fig, ax = plt.subplots(1, 2, figsize=(figsize, 0.35 * figsize), sharex=True, sharey=True)
    ax = ax.flat
    
    # plot prior
    sns.histplot(
        data=rt_prior, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=True, 
        stat=stat, label=f"Prior (KS={ks_prior:.3f})", ax=ax[0]
    )
    sns.histplot(
        data=rt_data, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=False, 
        stat=stat, label="Empirical", ax=ax[0]
    )

    ax[0].legend()
    ax[0].set_xlabel("BRT (s)")
    ax[0].set_ylabel("Cumulative density")

    # plot posterior
    sns.histplot(
        data=rt_post, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=True, 
        stat=stat, label=f"Posterior (KS={ks_post:.3f})", ax=ax[1]
    )
    sns.histplot(
        data=rt_data, binrange=binrange, cumulative=cumulative, 
        fill=False, element=hist_type, common_norm=False, 
        stat=stat, label="Empirical", ax=ax[1]
    )

    ax[1].legend()
    ax[1].set_xlabel("BRT (s)")
    ax[1].yaxis.set_visible(False)
    
    # custom axis
    ax[0].set_xticklabels(0.1 * ax[0].get_xticks().round(3))
    ax[1].set_xticklabels(0.1 * ax[1].get_xticks().round(3))

    plt.tight_layout()

    if show:
        plt.show()
    return fig

def plot_triangle_scatter_matrix(df, figsize=10, show=True):
    """ last columns is target by default """
    z_lim = np.ceil(np.abs(df.iloc[:, 0]).max())
    num_cols = len([c for c in df.columns if "Factor" in c]) - 1

    fig,ax = plt.subplots(
        num_cols, num_cols, figsize=(figsize,figsize), constrained_layout=True,
    )
    fig.add_gridspec(
        ncols=num_cols, nrows=num_cols, 
        width_ratios=[figsize/num_cols for i in range(num_cols)], 
        height_ratios=[figsize/num_cols for i in range(num_cols)]
    )
    for i in range(num_cols):
        for j in range(num_cols):
            if i < j:
                ax[i, j].xaxis.set_visible(False)
                ax[i, j].yaxis.set_visible(False)
                ax[i, j].axis("off")
            else:
                p = ax[i, j].scatter(
                    df.iloc[:,j], df.iloc[:,i+1], c=df.iloc[:,-1], s=1.5
                )
                
                # custom labels
                if i < num_cols - 1 and j == 0:
                    ax[i, j].set_ylabel(df.columns[i+1])
                    ax[i, j].xaxis.set_visible(False)
                elif i == num_cols - 1 and j > 0:
                    ax[i, j].set_xlabel(df.columns[j])
                    ax[i, j].yaxis.set_visible(False)
                elif i == num_cols - 1 and j == 0:
                    ax[i, j].set_xlabel(df.columns[j])
                    ax[i, j].set_ylabel(df.columns[i+1])
                else:
                    ax[i, j].xaxis.set_visible(False)
                    ax[i, j].yaxis.set_visible(False)

    cb = fig.colorbar(p, ax=ax[i, :], shrink=1, location='bottom', label=df.columns[-1])
    cb.set_label(label="BRT (s)")

    if show:
        plt.show()
    return fig

def plot_cf_scatter_horizontal(df, figsize=10, show=True):
    factor_y = 4
    factor_x = [1, 3]
    factor_cols = [c for c in df.columns if "Factor" in c]
    factor_names = ["", "", "", ""]
    delays = df["Delay"].unique()
    z_lim = np.ceil(np.abs(df.iloc[:, 0]).max())
    mid_col = np.ceil((len(delays) - 1) / 2).astype(int)

    fig, ax = plt.subplots(
        len(factor_x), len(delays), 
        figsize=(figsize, figsize / len(delays) * len(factor_x) * 1),
    )
    fig.add_gridspec(
        nrows=len(factor_x), 
        ncols=len(delays), 
        width_ratios=[1 for i in range(len(delays))],
        height_ratios=[1 for i in range(len(factor_x))]
    )
    
    for i in range(len(delays)):
        df_sub = df.loc[df["Delay"] == delays[i]]
        for j in range(len(factor_x)):
            p = ax[j, i].scatter(
                df_sub[f"Factor {factor_x[j]}"], 
                df_sub[f"Factor {factor_y}"], 
                c=df_sub["RT"], 
                cmap="viridis", s=1.5
            )

            ax[0, i].set_title(f"AEB delay = {delays[i]} (s)")

            # customize y axis
            if i == 0:
                ax[j, i].set_ylabel(f"Factor {factor_y}")
            elif i == len(delays) - 1: # facet factor name vertical last column
                ax[j, i].set_yticks([])
            else:
                ax[j, i].yaxis.set_visible(False)
            
            # customize x axis
            if j < len(factor_x) - 1: # remove x axis above last row
                if i == mid_col:
                    ax[j, i].set_xticks([])
                else:
                    ax[j,i].xaxis.set_visible(False)

            if i == mid_col:
                ax[j, i].set_xlabel(f"Factor {factor_x[j]}")
            
    plt.tight_layout(rect=[0, -0.08, 1, 1])
    cb = fig.colorbar(
        p, ax=ax.ravel().tolist(), 
        shrink=1, aspect=30, pad=0.15, 
        location='bottom', label="BRT (s)"
    )
    cb.set_label(label="Time-to-decision (s)")
    
    if show:
        plt.show()
    return fig 

def plot_cf_violin(df, figsize=6, show=False):
    df["Brake time category"] = np.zeros(len(df))
    df["Brake time category"].loc[df["RT"] > df["Delay"]] = "After AV"
    df["Brake time category"].loc[df["RT"] <= df["Delay"]] = "Before AV"
    df["Brake time category"].loc[df["RT"] == 2.9] = "Never"
    
    fig, ax = plt.subplots(1, 1, figsize=(figsize, 0.5 * figsize))
    sns.violinplot(
        data=df, x="Delay", y="RT",
        cut=0, scale="count", inner=None, color="silver", ax=ax
    )
    sns.stripplot(
        data=df, x="Delay", y="RT", hue="Brake time category", size=2
    )

    ax.set_ylabel("Driver time-to-decision (s)")
    ax.set_xlabel("AV BRT (s)")
    ax.legend(loc="lower left")
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def main(exp_path, diagnostics=False, dpi=100, save=False, show=False):
    # load results
    with open(os.path.join(exp_path, "tab", "learning_stats.p"), "rb") as f:
        stats_dict = pickle.load(f)
    
    with open(os.path.join(exp_path, "tab", "factor_selection.p"), "rb") as f:
        dict_selection = pickle.load(f)
        
    df_history = pd.read_csv(os.path.join(exp_path, "history.csv"))
    df_prior = pd.read_csv(os.path.join(exp_path, "tab", "prior.csv"))
    df_post = pd.read_csv(os.path.join(exp_path, "tab", "post.csv"))
    df_post_map = pd.read_csv(os.path.join(exp_path, "tab", "post_map.csv"))
    df_predictive = pd.read_csv(os.path.join(exp_path, "tab", "sample_predictive.csv"))
    df_pairwise_ks_stats = pd.read_csv(os.path.join(exp_path, "tab", "pairwise_ks.csv"))
    df_corr = pd.read_csv(os.path.join(exp_path, "tab", "params_corr.csv"), index_col=0)
    df_factor_loading = pd.read_csv(os.path.join(exp_path, "tab", "factor_loading.csv"))
    df_factor_assignment = pd.read_csv(os.path.join(exp_path, "tab", "factor_assignment.csv"))
    df_factor_samples = pd.read_csv(os.path.join(exp_path, "tab", "factor_samples.csv"))
    df_cf = pd.read_csv(os.path.join(exp_path, "tab", "cf_simulation.csv"))
    df_cf_sub = df_cf.loc[df_cf["Delay"].isin([0.5, 2.5, 4.5])]
    
    seed = 0
    num_factors = df_factor_loading.shape[0] - 1
    num_samples = 3000
    param_vars = [c for c in df_post.columns if "$" in c]
    
    # plot learning
    fig_history = plot_learning_curve(df_history, show=False)
    fig_stats_dist = plot_stats_dist(stats_dict, show=False)
    fig_sample_predictive = plot_predictive(df_predictive, cumulative=True, show=False)
    # plot params hist
    fig_prior_hist = plot_params_hist(df_prior[param_vars], show=False)
    fig_post_hist = plot_params_hist(
        df_post[param_vars].sample(n=num_samples, random_state=seed), show=False
    )
    fig_post_map_hist = plot_params_hist(df_post_map[param_vars], show=False)
    # plot params by var
    fig_params_by_scenario = plot_params_by_group(
        df_post, param_vars, "Scenario", figsize=7, show=False
    )
    fig_params_by_age = plot_params_by_group(
        df_post_map, param_vars, "Age", plot_type="scatter", figsize=6, show=False
    )
    fig_params_by_gender = plot_params_by_group(
        df_post_map, param_vars, "Gender", swarm=True, figsize=6, show=False
    )
    # plot params diagnostics
    fig_pariwise_ks = plot_params_hist(df_pairwise_ks_stats, show=False)
    fig_corr = plot_params_corr(df_corr, show=False)
    # plot factor
    fig_selection = plot_factor_selection(dict_selection, show=False)
    fig_loading = plot_factor_loadings(df_factor_loading, figsize=3.2, vertical=True, show=False)
    fig_factor_vs_rt = plot_factor_vs_rt(df_factor_assignment, show=False)
    fig_factor_by_scenario = plot_params_by_group(
        df_factor_assignment, 
        [c for c in df_factor_assignment.columns if "Factor"  in c], 
        "Alert", num_cols=num_factors, figsize=8, show=False
    )
    fig_factor_interpo_scatter = plot_triangle_scatter_matrix(
        df_factor_samples[[c for c in df_factor_samples.columns if "Factor" in c] + ["RT"]],
        figsize=5, show=False
    )
    
    # plot counterfactual
    fig_cf_scatter = plot_cf_scatter_horizontal(df_cf_sub, figsize=6, show=False)
    
    if diagnostics:
        # plot scatters
        num_samples = 1000
        fig_prior_scatter = plot_params_scatter(
            df_prior[param_vars].sample(n=num_samples, random_state=seed), show=False
        )
        fig_post_scatter = plot_params_scatter(
            df_post[param_vars].sample(n=num_samples), show=False
        )
        fig_post_map_scatter = plot_params_scatter(
            df_post_map[param_vars], show=False
        )
        fig_factor_scatter = plot_params_scatter(
            df_factor_samples[param_vars].sample(n=num_samples, random_state=seed), show=False
        )
    
    if save:
        fig_path = os.path.join(exp_path, "fig")
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        
        # learning plots
        fig_history.savefig(os.path.join(fig_path, "history.png"), dpi=dpi)
        fig_stats_dist.savefig(os.path.join(fig_path, "stats.png"), dpi=dpi)
        fig_sample_predictive.savefig(os.path.join(fig_path, "sample_predictive.png"), dpi=dpi)
        # params hist plots
        fig_prior_hist.savefig(os.path.join(fig_path, "prior_hist.png"), dpi=dpi)
        fig_post_hist.savefig(os.path.join(fig_path, "post_hist.png"), dpi=dpi)
        fig_post_map_hist.savefig(os.path.join(fig_path, "post_map_hist.png"), dpi=dpi)
        # params by var plots
        fig_params_by_scenario.savefig(os.path.join(fig_path, "params_by_scenario.png"), dpi=dpi)
        fig_params_by_age.savefig(os.path.join(fig_path, "params_by_age.png"), dpi=dpi)
        fig_params_by_gender.savefig(os.path.join(fig_path, "params_by_gender.png"), dpi=dpi)
        # params diagnostics plots
        fig_pariwise_ks.savefig(os.path.join(fig_path, "post_pairwise_ks_test.png"), dpi=dpi)
        fig_corr.savefig(os.path.join(fig_path, "post_map_corr.png"), dpi=dpi)
        # factor plots
        fig_selection.savefig(os.path.join(fig_path, "factor_selection.png"), dpi=dpi)
        fig_loading.savefig(os.path.join(fig_path, "factor_loading.png"), dpi=dpi)
        fig_factor_vs_rt.savefig(os.path.join(fig_path, "factor_vs_rt.png"), dpi=dpi)
        fig_factor_by_scenario.savefig(os.path.join(fig_path, "factor_by_scenario.png"), dpi=dpi)
        fig_factor_interpo_scatter.savefig(os.path.join(fig_path, "factor_interpo_scatter"), dpi=dpi)
        # counterfactual plots
        fig_cf_scatter.savefig(os.path.join(fig_path, "counterfactual_scatter.png"), dpi=dpi)
        
        if diagnostics:
            fig_prior_scatter.savefig(os.path.join(fig_path, "prior_scatter.png"), dpi=dpi)
            fig_post_scatter.savefig(os.path.join(fig_path, "post_scatter.png"), dpi=dpi)
            fig_post_map_scatter.savefig(os.path.join(fig_path, "post_map_scatter.png"), dpi=dpi)
            fig_factor_scatter.savefig(os.path.join(fig_path, "factor_samples_scatter.png"), dpi=dpi)
    
    print(f"figures saved={save}")
    
    if show:
        plt.show()
    plt.clf(); plt.close("all")

def parse_args():
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_path", type=str, default="../exp/ebirl", help="experiment path, default=../exp/ebirl"
    )
    parser.add_argument(
        "--exp_num", 
        type=str,
        default="lognormal_state_dim_2_embedding_dim_None_reward_ahead_"
            "True_prior_dist_mvn_num_components_3_post_dist_diag_iterative_True_init_"
            "uniform_True_obs_penalty_0.2_prior_lr_0.01_vi_lr_0.01_decay_0_seed_0", 
        help="experiment number, default=seed_0"
    )
    parser.add_argument("--diagnostics", type=bool_, default=False, help="perform diagnostics, defalt=False")
    parser.add_argument("--save", type=bool_, default=True, help="save figures, default=False")
    parser.add_argument("--show", type=bool_, default=False, help="show figures, default=True")
    arglist = parser.parse_args()
    return arglist

if __name__ == "__main__": 
    arglist = parse_args()
    result_path = os.path.join(arglist.exp_path, arglist.exp_num)
    
    main(
        result_path, 
        diagnostics=arglist.diagnostics, 
        save=arglist.save, 
        show=arglist.show
    )