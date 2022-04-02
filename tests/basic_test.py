import sys, os
sys.path.append('.')

from emlopt.search_loop import SearchLoop
from emlopt import solvers, surrogates
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext
from problems.simple_functions import polynomial

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
search.run()
