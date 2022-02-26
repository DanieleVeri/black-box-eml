import sys, os
sys.path.append('.')

from emlopt.search_loop import SearchLoop
from emlopt import solvers, surrogates
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

from problems.simple_functions import build_rosenbrock, mccormick, polynomial
from problems.ackley import build_ackley

wandb_cfg = WandbContext.get_defatult_cfg()

UNIFORM_NOISE = {
    "verbosity": 2,

    "iterations": 50,
    "starting_points": 3,

    "surrogate_model": {
        "type": surrogates.UniformNoise,
        "epochs": 999,
        "noise_ratio": 30,
        "sample_weights": 2,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": solvers.UCB,
        "lambda_ucb": 1,
        "solver_timeout": 60,
    }
}

EARLY_STOP = {
    "verbosity": 2,

    "iterations": 20,
    "starting_points": 3,

    "surrogate_model": {
        "type": surrogates.EarlyStop,
        "epochs": 999,
        "patience": 5,
        "num_val_points": 10,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": solvers.UCB,
        "lambda_ucb": 1,
        "solver_timeout": 60,
    }
}

STOP_CI = {
    "verbosity": 2,

    "iterations": 20,
    "starting_points": 3,

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
        "type": solvers.UCB,
        "lambda_ucb": 1,
        "solver_timeout": 60,
    }
}

DIST = {
    "verbosity": 2,

    "iterations": 20,
    "starting_points": 3,

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
        "lambda_ucb": 1,
        "solver_timeout": 120,
    }
}

CONFIG=DIST

NAME="polynomial_1d"
problem = build_problem(NAME, polynomial, ['real'], [[0,1]])
search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()

NAME="mccormick_2d"
problem = build_problem(NAME, mccormick, ['real', 'real'], [[-1.5, 4], [-3, 4]])
search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()

NAME="ackley_2d"
problem = build_problem(NAME, *build_ackley(2))
search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()

NAME="rosenbrock2d"
problem = build_problem(NAME, *build_rosenbrock(2))
search = SearchLoop(problem, CONFIG)
wandb_cfg['experiment_name'] = NAME
with WandbContext(wandb_cfg, search): search.run()
