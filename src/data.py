import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence 

def pad_collate(batch):
    """ Collate batch of dict to have the same sequence length """
    assert isinstance(batch[0], dict)
    keys = list(batch[0].keys())
    pad_batch = {k: pad_sequence([b[k] for b in batch]) for k in keys}
    mask = pad_sequence([torch.ones(len(b[keys[0]])) for b in batch])
    return pad_batch, mask

class DriverDataset(Dataset):
    def __init__(self, df, t_add=0):
        """ Custom driver dataset
        
        Args: 
            df (pd.dataframe): processed dataset
            t_add (int): additional timesteps to include after rt
        """
        super().__init__()
        self.df = df
        self.eps_ids = self.df["episode"].unique()

        # extract data
        obs = []
        act = []
        rt = []
        for e in self.eps_ids:
            df_eps = df.loc[df["episode"] == e].sort_values(by="t").reset_index(drop=True)
            rt_eps = df_eps["rt"].values[0] + 1 # fix this
            obs_eps = df_eps["obs"].values[:rt_eps + t_add]
            act_eps = df_eps["act"].values[:rt_eps + t_add]
            
            rt.append(rt_eps)
            obs.append(torch.from_numpy(obs_eps).to(torch.float32))
            act.append(torch.from_numpy(act_eps).long())
        
        self.rt = rt
        self.obs = obs
        self.act = act

    def __len__(self):
        return len(self.eps_ids)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        act = self.act[idx]
        return {"obs": obs, "act": act}