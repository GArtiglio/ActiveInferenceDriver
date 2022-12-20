import argparse
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# settings 
w = 1.9; l = 3 # car width, length
mph2ms = 0.44704 # speed unit conversion
brakepedal_threshold = 0.01

def calc_looming(D, v_rel):
    D = D - l
    theta = 2 * np.arctan(0.5 * w / D) #* 180 / np.pi
    theta_dot = w * v_rel / (D**2 + w**2/4)
    loom = theta_dot/theta
    return loom

def process_one_drive(df, horizon=30, debug=False):
    """ returns processed df with observation and action columns """
    df = df[df["Time"] >= df["start"].iloc[0]].reset_index()
    df = df.assign(speed_ms=df["velocitymph"] * mph2ms)
    df = df.assign(crash=df["crash"] == "yes")
    
    # calculate looming 
    df = df.assign(d=np.abs(df["locationXaheadveh"] - df["locationXsub"]))
    df = df.assign(v_rel=np.abs(df["speed_ms"] - df["aheadvehspeed"]))
    df = df.assign(obs=calc_looming(df["d"], df["v_rel"]))
    
    # get braking reaction 
    df = df.assign(
        is_braking=(df["brakepedalposition"] > brakepedal_threshold
    ).astype(int))
    rt = np.where(df["is_braking"]==1)[0][0]
    df = df.assign(rt=rt)
    df = df.iloc[:horizon, :]
    
    # get action sequence to horizon
    act = np.zeros(len(df))
    act[rt:] = 1
    df = df.assign(act=act)

    # add time index
    df = df.assign(t=np.arange(df.shape[0]))
    
    # select columns
    df = df[["Participant_ID", "Order", "Scenario", "Alert", "crash", "rt", "t", "obs", "act", "speed_ms", "d"]]
    
    # if debug:
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(3, 1, figsize=(4, 4))
    #     ax[0].plot(df["obs"], "-o")
    #     ax[0].set_ylabel("loom")
    #     ax[1].plot(df["brakepedalposition"], "o-")
    #     ax[1].set_ylabel("brake")
    #     ax[2].plot(df["act"], "o-")
    #     ax[2].set_ylabel("act")
    #     plt.tight_layout()
    #     plt.show()
    
    return df

def main(path, horizon, debug=False):
    data_path = os.path.join(path, "raw")
    save_path = os.path.join(path, "processed")
    
    file_paths = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"total num files: {len(file_paths)}")
    
    df_all_drives = []
    for i, file_path in enumerate(tqdm(file_paths)):
        df_drive = pd.read_csv(file_path, index_col=0)
        df_drive_processed = process_one_drive(df_drive, horizon=horizon, debug=debug)
        df_drive_processed = df_drive_processed.assign(episode=i)
        df_all_drives.append(df_drive_processed)
        
        if debug and i >= 10:
            break
    
    df_all_drives = pd.concat(df_all_drives)
    
    print("preprocess done")
    
    if not debug:
        df_all_drives.to_csv(os.path.join(save_path, "processed_data.csv"), index=False)
        print("data saved")
        
if __name__ == "__main__":
    bool_ = lambda x: x if isinstance(x, bool) else x == "True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../dat", help="data path, default=./dat")
    parser.add_argument("--horizon", type=int, default=100, help="episode horizon, default=100")
    parser.add_argument("--debug", type=bool_, default=False, help="debug, default=False")
    arglist = parser.parse_args()
    
    main(
        arglist.path,
        arglist.horizon, 
        arglist.debug,
    )