import sys, os
sys.path.append('.')


import numpy as np
from emlopt.search_loop import SearchLoop
from emlopt import solvers, surrogates
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

wandb_cfg = WandbContext.get_defatult_cfg()
problem = build_problem("fn", lambda x: np.random.random(), ['real']*10, [[0,1]]*10)

CONFIG = {
    "verbosity": 2,

    "iterations": 10,
    "starting_points": 100,

    "surrogate_model": {
        "type": surrogates.StopCI,
        "epochs": 999,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": solvers.SimpleDist,
        "lambda_ucb": 1000,
        "solver_timeout": 600,
    }
}

search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = "VANILLA"
with WandbContext(wandb_cfg, search): search.run()

CONFIG["milp_model"]["type"]= solvers.IncrementalDist

search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = "INCREMENTAL"
with WandbContext(wandb_cfg, search): search.run()