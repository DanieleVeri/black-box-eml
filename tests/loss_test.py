import sys, os
sys.path.append('.')

from emlopt.search_loop import SearchLoop
from emlopt import surrogates, solvers
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

from problems.simple_functions import build_rosenbrock, mccormick, polynomial
from problems.ackley import build_ackley

wandb_cfg = WandbContext.get_defatult_cfg()

LR_HIGH = {
    "verbosity": 2,

    "iterations": 1,
    "starting_points": 1000,

    "surrogate_model": {
        "type": surrogates.StopCI,
        "epochs": 999,
        "learning_rate": 5e-2,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": solvers.UCB,
        "lambda_ucb": 1,
        "solver_timeout": 120,
    }
}

LR_LOW = {
    "verbosity": 2,

    "iterations": 1,
    "starting_points": 1000,

    "surrogate_model": {
        "type": surrogates.StopCI,
        "epochs": 999,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": solvers.UCB,
        "lambda_ucb": 1,
        "solver_timeout": 120,
    }
}


# NAME="mccormick_2d_highlr"
# problem = build_problem(NAME, mccormick, ['real', 'real'], [[-1.5, 4], [-3, 4]])
# search = SearchLoop(problem, LR_HIGH)
# wandb_cfg['experiment_name'] = NAME
# with WandbContext(wandb_cfg, search): search.run()

# NAME="mccormick_2d_lowlr"
# problem = build_problem(NAME, mccormick, ['real', 'real'], [[-1.5, 4], [-3, 4]])
# search = SearchLoop(problem, LR_LOW)
# wandb_cfg['experiment_name'] = NAME
# with WandbContext(wandb_cfg, search): search.run()


NAME="ackley_2dhighlr"
problem = build_problem(NAME, *build_ackley(2))
search = SearchLoop(problem, LR_HIGH)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()

NAME="ackley_2dlowlr"
problem = build_problem(NAME, *build_ackley(2))
search = SearchLoop(problem, LR_LOW)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()