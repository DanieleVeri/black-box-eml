import sys
sys.path.append('..')

import numpy as np
import unittest
from base_test import BaseTest
from emlopt import solvers, surrogates
from emlopt.utils import set_seed
from emlopt.problem import build_problem
from experiments.problems.simple_functions import polynomial, build_rosenbrock, mccormick

CONFIG = {
    "equals_delta": 1e-6,
    "verbosity": 2,
    "starting_points": 10,
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

class EMLBackendTest(BaseTest):

    @classmethod
    def setUpClass(cls):
        super(EMLBackendTest, cls).setUpClass()
        def linear_constraint(backend, model, xvars):
            return [[xvars[0] - xvars[1] <= 0, "ineq"]]
        cls.rosenbrock = build_problem("rosenbrock_3D", *build_rosenbrock(3), constraint_cb=linear_constraint)
        cls.polynomial = build_problem("polynomial_1D", polynomial, ['real'], [[0,1]])
        cls.mccormick = build_problem("mccormick_int_2D", mccormick, ['int', 'int'], [[-2, 2], [-2, 2]])

        cls.dataset_rosenbrock = cls.rosenbrock.get_dataset(CONFIG["starting_points"], backend_type='cplex')
        cls.dataset_polynomial = cls.polynomial.get_dataset(CONFIG["starting_points"])
        cls.dataset_mccormick = cls.mccormick.get_dataset(CONFIG["starting_points"])

        surrogate_model = surrogates.StopCI(cls.rosenbrock, CONFIG['surrogate_model'], cls.test_logger)
        cls.model_rosenbrock, _ = surrogate_model.fit_surrogate(*cls.dataset_rosenbrock, timer_logger=cls.test_logger)
        surrogate_model = surrogates.StopCI(cls.polynomial, CONFIG['surrogate_model'], cls.test_logger)
        cls.model_polynomial, _ = surrogate_model.fit_surrogate(*cls.dataset_polynomial, timer_logger=cls.test_logger)
        surrogate_model = surrogates.StopCI(cls.mccormick, CONFIG['surrogate_model'], cls.test_logger)
        cls.model_mccormick, _ = surrogate_model.fit_surrogate(*cls.dataset_mccormick, timer_logger=cls.test_logger)

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

    def test_cplex_polynomial1D_surrogate_match_solver(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.polynomial, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_polynomial, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_polynomial, *self.dataset_polynomial, timer_logger=self.test_logger)

    def test_ortools_polynomial1D_surrogate_match_solver(self):
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.polynomial, cfg, 1, self.test_logger)
        ortools_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_polynomial, ortools_milp_model)
        ortools_milp_model.optimize_acquisition_function(self.model_polynomial, *self.dataset_polynomial, timer_logger=self.test_logger)

    def test_cplex_rosenbrock3D_surrogate_match_solver(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.rosenbrock, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, cplex_milp_model)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    def test_ortools_rosenbrock3D_surrogate_match_solver(self):
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.rosenbrock, cfg, 1, self.test_logger)
        ortools_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_rosenbrock, ortools_milp_model)
        ortools_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)

    def test_cplex_mccormick_integer_2D_surrogate_match_solver(self):
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.mccormick, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_mccormick, cplex_milp_model)
        opt_x, _ = cplex_milp_model.optimize_acquisition_function(self.model_mccormick, *self.dataset_mccormick, timer_logger=self.test_logger)
        self.assertAlmostEqual(opt_x[0], np.round(opt_x[0]), delta=CONFIG['equals_delta'])
        self.assertAlmostEqual(opt_x[1], np.round(opt_x[1]), delta=CONFIG['equals_delta'])

    def test_ortools_mccormick_integer_2D_surrogate_match_solver(self):
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.mccormick, cfg, 1, self.test_logger)
        ortools_milp_model.solution_callback = self._check_surrogate_match_solver(self.model_mccormick, ortools_milp_model)
        opt_x, _ = ortools_milp_model.optimize_acquisition_function(self.model_mccormick, *self.dataset_mccormick, timer_logger=self.test_logger)
        self.assertAlmostEqual(opt_x[0], np.round(opt_x[0]), delta=CONFIG['equals_delta'])
        self.assertAlmostEqual(opt_x[1], np.round(opt_x[1]), delta=CONFIG['equals_delta'])

    def test_rosenbrock3D_cross_backend_match(self):
        results = {
            'ortools': None,
            'cplex': None
        }
        def update_results(backend):
            def cb(main_variables, all_variables):
                results[backend] = all_variables['vars']
            return cb
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.rosenbrock, cfg, 1, self.test_logger)
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.rosenbrock, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = update_results('cplex')
        ortools_milp_model.solution_callback = update_results('ortools')
        ortools_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        cplex_milp_model.optimize_acquisition_function(self.model_rosenbrock, *self.dataset_rosenbrock, timer_logger=self.test_logger)
        for k,v in results['ortools'].items():
            if 'nn_z' not in k: # the activation can change value due to a floating point error
                self.assertAlmostEqual(results['ortools'][k], results['cplex'][k], delta=CONFIG["equals_delta"])

    def test_polynomial1D_cross_backend_match(self):
        results = {
            'ortools': None,
            'cplex': None
        }
        def update_results(backend):
            def cb(main_variables, all_variables):
                results[backend] = all_variables['vars']
            return cb
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.polynomial, cfg, 1, self.test_logger)
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.polynomial, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = update_results('cplex')
        ortools_milp_model.solution_callback = update_results('ortools')
        ortools_milp_model.optimize_acquisition_function(self.model_polynomial, *self.dataset_polynomial, timer_logger=self.test_logger)
        cplex_milp_model.optimize_acquisition_function(self.model_polynomial, *self.dataset_polynomial, timer_logger=self.test_logger)
        for k,v in results['ortools'].items():
            self.assertAlmostEqual(results['ortools'][k], results['cplex'][k], delta=CONFIG["equals_delta"])

    def test_integer_mccormick2D_cross_backend_match(self):
        results = {
            'ortools': None,
            'cplex': None
        }
        def update_results(backend):
            def cb(main_variables, all_variables):
                results[backend] = all_variables['vars']
            return cb
        cfg = {"backend": 'cplex', "lambda_ucb": 1, "solver_timeout": 30}
        cplex_milp_model = solvers.SimpleDist(self.mccormick, cfg, 1, self.test_logger)
        cfg = {"backend": 'ortools', "lambda_ucb": 1, "solver_timeout": 30}
        ortools_milp_model = solvers.SimpleDist(self.mccormick, cfg, 1, self.test_logger)
        cplex_milp_model.solution_callback = update_results('cplex')
        ortools_milp_model.solution_callback = update_results('ortools')
        ortools_milp_model.optimize_acquisition_function(self.model_mccormick, *self.dataset_mccormick, timer_logger=self.test_logger)
        cplex_milp_model.optimize_acquisition_function(self.model_mccormick, *self.dataset_mccormick, timer_logger=self.test_logger)
        for k,v in results['ortools'].items():
            self.assertAlmostEqual(results['ortools'][k], results['cplex'][k], delta=CONFIG["equals_delta"])

if __name__ == '__main__':
    unittest.main()
