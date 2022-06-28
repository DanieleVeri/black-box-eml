import sys
sys.path.append('.')

import numpy as np
import unittest
from base_test import BaseTest
from emlopt.search_loop import SearchLoop
from emlopt import solvers, surrogates
from emlopt.utils import set_seed
from emlopt.problem import build_problem
from experiments.problems.simple_functions import polynomial, build_rosenbrock, mccormick

CONFIG = {
    "equals_delta": 1e-6,
    "verbosity": 2,
    "starting_points": 1000
}

class SurrogateMethodTest(BaseTest):

    @classmethod
    def setUpClass(cls):
        super(SurrogateMethodTest, cls).setUpClass()
        def linear_constraint(backend, model, xvars):
            return [[xvars[0] - xvars[1] <= 0, "ineq"]]
        cls.rosenbrock = build_problem("rosenbrock_3D", *build_rosenbrock(3), constraint_cb=linear_constraint)
        cls.dataset_rosenbrock = cls.rosenbrock.get_dataset(CONFIG["starting_points"], backend_type='cplex')

    def test_stop_ci(self):
        surrogate_cfg = {
            "epochs": 10,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "batch_size": None,
            "depth": 2,
            "width": 20,
            "ci_threshold": 5e-2,
        }
        surrogate_model = surrogates.StopCI(self.rosenbrock, surrogate_cfg, self.test_logger)
        self.model_rosenbrock, _ = surrogate_model.fit_surrogate(*self.dataset_rosenbrock, timer_logger=self.test_logger)

    def test_early_stop(self):
        surrogate_cfg = {
            "epochs": 10,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "batch_size": None,
            "depth": 2,
            "width": 20,
            "num_val_points": 20,
            "patience": 5,
            "backend": "cplex"
        }
        surrogate_model = surrogates.EarlyStop(self.rosenbrock, surrogate_cfg, self.test_logger)
        self.model_rosenbrock, _ = surrogate_model.fit_surrogate(*self.dataset_rosenbrock, timer_logger=self.test_logger)

    def test_uniform_noise(self):
        surrogate_cfg = {
            "epochs": 10,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "batch_size": None,
            "depth": 2,
            "width": 20,
            "noise_ratio": 2,
            "sample_weights": 1,
            "ci_threshold": 5e-2,
            "backend": "cplex"
        }
        surrogate_model = surrogates.UniformNoise(self.rosenbrock, surrogate_cfg, self.test_logger)
        self.model_rosenbrock, _ = surrogate_model.fit_surrogate(*self.dataset_rosenbrock, timer_logger=self.test_logger)

if __name__ == '__main__':
    unittest.main()
