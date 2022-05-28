import sys, os
import unittest
import logging
from test_utils import create_logger
import numpy as np
sys.path.append('.')
from emlopt import solvers, surrogates
from emlopt.emllib.backend import get_backend
from emlopt.utils import set_seed
from emlopt.problem import build_problem
from experiments.problems.simple_functions import build_rosenbrock
from emlopt.emllib.net.reader.keras_reader import read_keras_probabilistic_sequential


CONFIG = {
    "equals_delta": 1e-6,
    "test_mask": [1,1,1,1,1,1,1,1,1],
    "verbosity": 2,
    "starting_points": 100,
    "surrogate_model": {
        "epochs": 10,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 2,
        "width": 20,
        "ci_threshold": 5e-2,
    }
}

class EMLBackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(EMLBackendTest, cls).setUpClass()
        cls.test_logger = create_logger('emllib-test')
        cls.test_logger.setLevel(logging.DEBUG)
        set_seed()

        def linear_constraint(backend, model, xvars):
            return [[xvars[0] - xvars[1] <= 0, "ineq"]]
        cls.rosenbrock = build_problem("rosenbrock_3D", *build_rosenbrock(3), constraint_cb=linear_constraint)
        cls.dataset_rosenbrock = cls.rosenbrock.get_dataset(CONFIG["starting_points"], backend_type='cplex')
        surrogate_model = surrogates.StopCI(cls.rosenbrock, CONFIG['surrogate_model'], cls.test_logger)
        cls.model_rosenbrock, _ = surrogate_model.fit_surrogate(*cls.dataset_rosenbrock, timer_logger=cls.test_logger)

    def setUp(self):
        set_seed()

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
    def test_milp_tighter_than_ibr(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'ibr'}
        ibr_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'milp'}
        milp_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        parsed_mdl = read_keras_probabilistic_sequential(self.model_rosenbrock)
        ibr_bounded, _ = ibr_solver.propagate_bound(parsed_mdl, timer_logger=self.test_logger)
        parsed_mdl = read_keras_probabilistic_sequential(self.model_rosenbrock)
        milp_bounded, _ = milp_solver.propagate_bound(parsed_mdl, timer_logger=self.test_logger)
        ibr_last_bounds = np.stack((ibr_bounded.layer(-1).ylb(), ibr_bounded.layer(-1).yub()))
        milp_last_bounds = np.stack((milp_bounded.layer(-1).ylb(), milp_bounded.layer(-1).yub()))
        self.assertTrue(milp_last_bounds[0,0] >= ibr_last_bounds[0,0])  # mu lb
        self.assertTrue(milp_last_bounds[1,0] <= ibr_last_bounds[1,0])  # mu ub
        self.assertTrue(milp_last_bounds[0,1] >= ibr_last_bounds[0,1])  # logstddev lb
        self.assertTrue(milp_last_bounds[1,1] <= ibr_last_bounds[1,1])  # logstddev ub

    @unittest.skipIf(CONFIG['test_mask'][1]==0, "skip")
    def test_milp_match_both_propagation_cross_backend(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'milp'}
        milp_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'both'}
        both_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        parsed_mdl = read_keras_probabilistic_sequential(self.model_rosenbrock)
        milp_bounded, _ = milp_solver.propagate_bound(parsed_mdl, timer_logger=self.test_logger)
        parsed_mdl = read_keras_probabilistic_sequential(self.model_rosenbrock)
        both_bounded, _ = both_solver.propagate_bound(parsed_mdl, timer_logger=self.test_logger)
        self.assertEqual(str(milp_bounded), str(both_bounded))

    @unittest.skipIf(CONFIG['test_mask'][2]==0, "skip")
    def test_tight_bound_propagation_full_cross_backend(self):
        results = {
            'milp': None,
            'both': None
        }
        def update_results(backend):
            def cb(main_variables, all_variables):
                results[backend] = main_variables
            return cb
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'milp'}
        milp_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30, "bound_propagation": 'both'}
        both_solver = solvers.UCB(self.rosenbrock, cfg, 1, self.test_logger)
        milp_solver.solution_callback = update_results('milp')
        both_solver.solution_callback = update_results('both')
        milp_solver.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        both_solver.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        for k,v in results['milp'].items():
            self.assertAlmostEqual(results['milp'][k], results['both'][k], delta=CONFIG["equals_delta"])


if __name__ == '__main__':
    unittest.main()
