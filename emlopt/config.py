from .surrogates import StopCI
from .solvers import IncrementalDist


DEFAULT = {
    "verbosity": 2,

    "iterations": 100,
    "starting_points": 10,

    "surrogate_model": {
        "type": StopCI,
        "epochs": 999,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": IncrementalDist,
        "lambda_ucb": 1,
        "solver_timeout": 60,
        "lns_fixed": 1
    }
}