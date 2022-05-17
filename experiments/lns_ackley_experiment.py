import sys, os
sys.path.append('.')

from emlopt.search_loop import SearchLoop
from emlopt import surrogates, solvers
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext
from problems.simple_functions import build_ackley

CONFIG = {
    "verbosity": 2,

    "iterations": 100,
    "starting_points": 10,

    "surrogate_model": {
        "type": "stop_ci",
        "epochs": 999,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": "lns_dist",
        "backend": "cplex",
        "lambda_ucb": None,
        "solver_timeout": 120,
        "sub_problems": 3
    }
}

def constraint_scbo(cplex, xvars):
    r = 5
    acc1 = 0
    for xi in xvars:
        acc1 += xi*xi
    acc2 = 0
    for xi in xvars:
        acc2 += xi
    return [[acc1 <= r*r, "center_dist"], [acc2 <= 0, "cst2"]]

problem = build_problem("ackley_10d_cst", *build_ackley(10), constraint_scbo)
search = SearchLoop(problem, CONFIG)

wandb_cfg = WandbContext.get_defatult_cfg()
wandb_cfg['experiment_name'] = "LNS"

with WandbContext(wandb_cfg, search):
    search.run()
