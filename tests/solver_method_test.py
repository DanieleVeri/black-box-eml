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

from experiments.problems.simple_functions import polynomial, build_rosenbrock, mccormick

CONFIG = {
    "equals_delta": 1e-6,
    "test_mask": [1,1,1,1,1,1,1,1,1],
    "verbosity": 2,
    "starting_points": 10
}

class SolverMethodTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(SolverMethodTest, cls).setUpClass()
        cls.test_logger = create_logger('emllib-test')
        cls.test_logger.setLevel(logging.DEBUG)
        set_seed()
        cls.rosenbrock = build_problem("rosenbrock_3D", *build_rosenbrock(3))
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

    ''' Test callback that given the solution, check if the surrogate prediction match the solver one'''
    def _check_surrogate_match_solver(self, learned_model, milp_model):
        def check_surrogate_match_solver(main_variables, all_variables):
            sol = np.expand_dims(milp_model.extract_solution(all_variables['vars'], scaled=True), 0)
            nn_prediction = learned_model(sol)
            nn_mean = nn_prediction.mean().numpy()[0,0]
            nn_stddev = nn_prediction.stddev().numpy()[0,0]
            solver_mean = all_variables['vars']['out_mean']
            solver_stddev = np.exp(all_variables['vars']['out_std'])
            self.test_logger.debug(f"Predicted mean {nn_mean}")
            self.test_logger.debug(f"Predicted stddev {nn_stddev}")
            self.assertAlmostEqual(nn_mean, solver_mean, delta=CONFIG["equals_delta"])
            self.assertAlmostEqual(nn_stddev, solver_stddev, delta=CONFIG["equals_delta"])
        return check_surrogate_match_solver

    @unittest.skipIf(CONFIG['test_mask'][0]==0, "skip")
    def test_UCB_surrogate_match_solver(self):
        solver_cfg = {
            'backend': "ortools",
            "lambda_ucb": 1,
            "solver_timeout": 30,
        }
        cplex_milp_model = solvers.UCB(self.rosenbrock, solver_cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    @unittest.skipIf(CONFIG['test_mask'][1]==0, "skip")
    def test_simple_dist_surrogate_match_solver(self):
        solver_cfg = {
            'backend': "ortools",
            "lambda_ucb": 1,
            "solver_timeout": 30,
        }
        cplex_milp_model = solvers.SimpleDist(self.rosenbrock, solver_cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    @unittest.skipIf(CONFIG['test_mask'][2]==0, "skip")
    def test_dynamic_lambda_surrogate_match_solver(self):
        solver_cfg = {
            'backend': "ortools",
            "lambda_ucb": None,
            "solver_timeout": 30,
        }
        cplex_milp_model = solvers.SimpleDist(self.rosenbrock, solver_cfg, 1, self.test_logger)
        cplex_milp_model.current_iteration = 0
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    @unittest.skipIf(CONFIG['test_mask'][3]==0, "skip")
    def test_incremental_dist_surrogate_match_solver(self):
        solver_cfg = {
            'backend': "ortools",
            "lambda_ucb": 1,
            "solver_timeout": 30,
        }
        cplex_milp_model = solvers.IncrementalDist(self.rosenbrock, solver_cfg, 1, self.test_logger)
        cplex_milp_model.current_iteration = 0
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    @unittest.skipIf(CONFIG['test_mask'][4]==0, "skip")
    def test_lns_dist_surrogate_match_solver(self):
        solver_cfg = {
            "backend": "ortools",
            "lambda_ucb": 1,
            "sub_problems": 2,
            "solver_timeout": 30,
        }
        cplex_milp_model = solvers.LNSDist(self.rosenbrock, solver_cfg, 1, self.test_logger)
        cplex_milp_model.current_iteration = 0
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)


if __name__ == '__main__':
    unittest.main()
