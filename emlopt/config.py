from . import solvers, surrogates


DEFAULT = {
    "verbosity": 2,

    "iterations": 100,
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
        "type": solvers.DynamicLambdaDist,
        "lambda_ucb": 1,
        "solver_timeout": 600,
    }
}

INCREMENTAL_TEST = {
    "verbosity": 2,

    "iterations": 100,
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
        "type": solvers.IncrementalDist,
        "lambda_ucb": 1,
        "solver_timeout": 60,
    }
}

UNIFORM_NOISE = {
    "verbosity": 2,

    "iterations": 100,
    "starting_points": 10,

    "surrogate_model": {
        "type": surrogates.UniformNoise,
        "epochs": 999,
        "noise_ratio": 10,
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

    "iterations": 100,
    "starting_points": 10,

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