import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option('display.max_rows', 500)

def main(data_path, save_path, min_xpos, max_xpos, freq, act_type, exclude_drive_id, save):
    """
    Args:
        data_path (str): path contain all data .csv files
        save_path (str): path to save a single pickle data file
        min_xpos (float): minimum x coordinate to cut off drive data
        max_xpos (float): maximum x coordinate to cut off drive data
        freq (int): down sample frequency. freq=1 corresponds to no downsample. 
            original frequency is 10 Hz
        act_type (str): action type. engage_state uses engagement state as action (2 actions). 
            engage_change uses engagement change as action (3 actions).
        exclude_drive_id (list): drive ids to exclude
        save (bool): whether to save processed data
    """
    file_paths = glob.glob(os.path.join(data_path, "*.csv"))
    
    data = []
    for f in tqdm(file_paths):
        drive_id = int(os.path.basename(f).split("_")[0])
        
        df_drive = pd.read_csv(f)
        df_drive = df_drive.assign(drive_id=int(drive_id))
        df_drive = df_drive.iloc[::freq, :] # down sample

        # reset obstacle pos
        obstacle_xpos = -4618.
        obstacle_ypos = -1194.6
        df_drive = df_drive.assign(Obstacle2XPos=obstacle_xpos)
        df_drive = df_drive.assign(Obstacle2YPos=obstacle_ypos)
        
        # recompute euclidean distance
        df_drive = df_drive.assign(
            euc_distance=np.sqrt((df_drive["XPos"].values - obstacle_xpos)**2 + \
                (df_drive["YPos"].values - obstacle_ypos)**2)
        )

        # offset initial and final pos
        df_drive = df_drive.loc[df_drive["XPos"] >= min_xpos]
        df_drive = df_drive.loc[df_drive["XPos"] <= max_xpos]
        df_drive = df_drive.assign(engage_act=df_drive["SimDriverEngaged"].diff().fillna(0))

        if act_type == "engage_state":
            df_drive = df_drive.assign(act=df_drive["SimDriverEngaged"])
        else:
            df_drive = df_drive.assign(act=df_drive["engage_act"])
        
        # compute time since sim driver state change
        idx_engage_act = list(np.where(df_drive["engage_act"] != 0)[0] + 1)
        idx_engage_act = [0] + idx_engage_act + [len(df_drive)]
        
        time_since_act = []    
        for i in range(len(idx_engage_act) - 1):
            counter = np.arange(idx_engage_act[i+1] - idx_engage_act[i])
            time_since_act.append(counter)

        df_drive = df_drive.assign(time_since_act=np.hstack(time_since_act))
        
        # handle empty files
        if len(df_drive) > 0 and drive_id not in exclude_drive_id:
            # remove initial disengage (handle drive 18)
            t_first_engaged = np.where(df_drive["SimDriverEngaged"] == 1)[0][0]
            df_drive = df_drive.iloc[t_first_engaged:]
        
            df_drive = df_drive.sort_values(by="Time").reset_index(drop=True)
            
            # collect data
            mdp_data = {
                "drive_id": drive_id,
                "t": df_drive["Time"].to_numpy().astype(np.float32),
                "obs": df_drive[["euc_distance", "time_since_act"]].to_numpy().astype(np.float32),
                "act": df_drive["act"].to_numpy().astype(int),
            }
            data.append(mdp_data)
    
    print(f"preprocess done, total drives {len(data)}")
    
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, "data.p"), "wb") as f:
            pickle.dump(data, f)

        print(f"data saved at {save_path}")
        
if __name__ == "__main__":
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../Scenario_2_data", help="data path, default=../../Scenario_2_data")
    parser.add_argument("--save_path", type=str, default="../../data", help="data path, default=../../data")
    parser.add_argument("--min_xpos", type=float, default=-4761., help="min x position, default=-4761.")
    parser.add_argument("--max_xpos", type=float, default=-4470., help="max x position, default=-4470.")
    parser.add_argument("--freq", type=int, default=5, help="data sampling frequency, freq=1 correspond to 10Hz, default=5")
    parser.add_argument("--act_type", type=str, choices=["engage_state", "engage_change"], default="engage_state")
    parser.add_argument("--exclude_drive_id", type=list, default=[18])
    parser.add_argument("--save", type=bool_, default=True)
    arglist = parser.parse_args()
    
    main(
        arglist.data_path,
        arglist.save_path,
        arglist.min_xpos,
        arglist.max_xpos, 
        arglist.freq,
        arglist.act_type,
        arglist.exclude_drive_id,
        arglist.save,
    )