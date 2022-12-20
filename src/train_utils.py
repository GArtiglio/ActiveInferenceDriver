import os
import json
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch

def train(model, loader, loss_fn, optimizer, epochs, verbose=1, callback=None, grad_check=False):
    history = []
    start_time = time.time()
    for e in range(epochs):
        pad_batch, mask = next(iter(loader))
        loss, stats = loss_fn(model, pad_batch["obs"] + 1e-6, pad_batch["act"], mask)
        loss.backward()
        
        if grad_check:
            for n, p in model.named_parameters():
                if p.grad is None:
                    print(n, p.requires_grad, None)
                else:
                    print(n, p.requires_grad, p.grad.data.norm())
            break
        
        optimizer.step()
        optimizer.zero_grad()
        
        tnow = time.time() - start_time
        stats.update({"time": tnow})
        history.append(stats)
        
        if (e + 1) % verbose == 0:
            print(f"e: {e+1}/{epochs}", model.get_stdout(stats))

        if callback is not None:
            callback(model, optimizer, stats)

    df_history = pd.DataFrame(history)
    return model, optimizer, df_history, callback

def plot_history(df_history, plot_keys=None, figsize=(12, 4)):
    if plot_keys is None:
        plot_keys = df_history.columns

    fig, ax = plt.subplots(1, len(plot_keys), figsize=figsize)
    for i, k in enumerate(plot_keys):
        ax[i].plot(df_history[k])
        ax[i].set_xlabel("epoch")
        ax[i].set_title(k)
    plt.tight_layout()
    return fig, ax


class Logger:
    def __init__(self, arglist, plot_keys, cp_history=None):
        date_time = datetime.datetime.now().strftime("%m-%d-%Y %H-%M-%S")
        self.save_path = os.path.join(arglist.exp_path, date_time)
        self.model_path = os.path.join(self.save_path, "models")

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # save args
        with open(os.path.join(self.save_path, "args.json"), "w") as f:
            json.dump(vars(arglist), f)

        self.cp_every = arglist.cp_every
        self.plot_keys = plot_keys
        self.history = []
        self.cp_history = cp_history
        self.iter = 0

        print(f"checkpoint: {date_time} created \n")

    def __call__(self, model, optimizer, stats):
        self.iter += 1
        self.history.append(stats)

        if self.iter % self.cp_every == 0:
            df_history = pd.DataFrame(self.history)
            self.save_history(df_history)
            self.save_checkpoint(model, optimizer, os.path.join(self.model_path, f"model_{self.iter}.pt"))

    def save_history(self, df_history):
        if self.cp_history is not None:
            df_history["epoch"] += self.cp_history["epoch"].values[-1] + 1
            df_history["time"] += self.cp_history["time"].values[-1]
            df_history = pd.concat([self.cp_history, df_history], axis=0)

        df_history.to_csv(os.path.join(self.save_path, "history.csv"), index=False)
        fig, _ = plot_history(df_history, plot_keys=self.plot_keys)
        fig.savefig(os.path.join(self.save_path, "history.png"), dpi=100)
        
        plt.clf()
        plt.close()
    
    def save_checkpoint(self, model, optimizer, path=None):
        if path is None:
            path = os.path.join(self.save_path, "model.pt")

        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        optimizer_state_dict = {
            k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in optimizer.state_dict().items()
        }
        
        torch.save({
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict
        }, path)
        print(f"\ncheckpoint saved at: {path}\n")