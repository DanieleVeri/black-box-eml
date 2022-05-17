import sys, os
import unittest
import logging
from test_utils import create_logger
import numpy as np
sys.path.append('.')
from emlopt.search_loop import SearchLoop
from emlopt import solvers, surrogates
from emlopt.utils import set_seed
from emlopt.problem import build_problem

from experiments.problems.simple_functions import build_rosenbrock

CONFIG = {
    "equals_delta": 1e-6,
    "test_mask": [1,1,1,1,1,1,1,1,1],
    "verbosity": 2,
    "starting_points": 50
}

class DistMethodsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(DistMethodsTest, cls).setUpClass()
        cls.test_logger = create_logger('emllib-test')
        cls.test_logger.setLevel(logging.DEBUG)
        set_seed()
        cls.rosenbrock = build_problem("rosenbrock_5D", *build_rosenbrock(5))
        cls.dataset_rosenbrock = cls.rosenbrock.get_dataset(CONFIG["starting_points"])
        surrogate_cfg = {
            "epochs": 10,
            "learning_rate": 5e-3,
            "weight_decay": 1e-4,
            "batch_size": None,
            "depth": 2,
            "width": 20,
            "ci_threshold": 5e-2,
        }
        surrogate_model = surrogates.StopCI(cls.rosenbrock, surrogate_cfg, cls.test_logger)
        cls.model_rosenbrock, _ = surrogate_model.fit_surrogate(*cls.dataset_rosenbrock, timer_logger=cls.test_logger)

        solver_cfg = {"backend": 'cplex', "lambda_ucb": 100, "solver_timeout": 30}
        simple_dist_model = solvers.SimpleDist(cls.rosenbrock, solver_cfg, 1, cls.test_logger)
        cls.opt_x, _ = simple_dist_model.optimize_acquisition_function(cls.model_rosenbrock, *cls.dataset_rosenbrock, timer_logger=cls.test_logger)

    @unittest.skipIf(CONFIG['test_mask'][0]==0, "skip")
    def test_incremental_dist_match(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 100, "solver_timeout": 30}
        incremental_dist_model = solvers.IncrementalDist(self.rosenbrock, cfg, 1, self.test_logger)
        opt_x, _ = incremental_dist_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        diff = self.opt_x - opt_x
        diff = np.sum(np.abs(diff))
        self.assertAlmostEqual(diff, 0.0, delta=CONFIG['equals_delta'])

    @unittest.skipIf(CONFIG['test_mask'][1]==0, "skip")
    def test_speedup_dist_match(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 100, "solver_timeout": 30}
        incremental_dist_model = solvers.SpeedupDist(self.rosenbrock, cfg, 1, self.test_logger)
        opt_x, _ = incremental_dist_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        diff = self.opt_x - opt_x
        diff = np.sum(np.abs(diff))
        self.assertAlmostEqual(diff, 0.0, delta=CONFIG['equals_delta'])

if __name__ == '__main__':
    unittest.main()
