from sklearn.model_selection import ParameterGrid
from train import * 

def grid_search(arglist, param_grid):
    print(f"\ngrid search: num grids: {len(param_grid)}..")
    for i, params in enumerate(param_grid):
        print(i, params)
        arglist.seed = int(params["seed"])
        arglist.obs_penalty = params["obs_penalty"]
        arglist.prior_dist = params["prior_dist"]
        arglist.num_components = params["num_components"]
        arglist.init_uniform = params["init_uniform"]
        arglist.prior_decay = params["prior_decay"]
        arglist.vi_decay = params["vi_decay"]
        arglist.prior_lr = params["prior_lr"]
        arglist.vi_lr = params["vi_lr"]
        arglist.epochs = params["epochs"]
        arglist.save = params["save"]
        
        train(arglist)
        
if __name__ == "__main__":
    arglist = parse_args()
    
    param_grid = ParameterGrid({
        "seed": [0],
        "obs_penalty": [1],
        "prior_dist": ["mvn"],
        "num_components": [3],
        "init_uniform": [True], 
        "prior_decay": [0],
        "vi_decay": [0],
        "prior_lr": [0.01], 
        "vi_lr": [0.01],
        "epochs": [10],
        "save": [False]
    })
    grid_search(arglist, param_grid)