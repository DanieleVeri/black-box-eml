import sys, os
import unittest
sys.path.append('.')
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
from emlopt.utils import set_seed
from experiments.problems.simple_functions import build_rosenbrock, mccormick, polynomial, build_ackley


CONFIG = {
    "verbosity": 2,

    "iterations": 20,
    "starting_points": 3,

    "surrogate_model": {
        "type": "stop_ci",
        "epochs": 999,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model": {
        "type": "simple_dist",
        "backend": "cplex",
        "lambda_ucb": 1,
        "solver_timeout": 120,
    }
}

class SearchConvergenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(SearchConvergenceTest, cls).setUpClass()
        set_seed()

    def test_simple_dist_polynomial_1d(self):
        problem = build_problem("polynomial_1d", polynomial, ['real'], [[0,1]])
        search = SearchLoop(problem, CONFIG)
        opt_X, opt_y = search.run()
        self.assertAlmostEqual(opt_y, 0.618, delta=1e-3)

    def test_simple_dist_mccormick_2d(self):
        problem = build_problem("mccormick_2d", mccormick, ['real', 'real'], [[-1.5, 4], [-3, 4]])
        search = SearchLoop(problem, CONFIG)
        opt_X, opt_y = search.run()
        self.assertAlmostEqual(opt_y, -1.913, delta=1e-3)

    def test_simple_dist_ackley_2d(self):
        problem = build_problem("ackley_2d", *build_ackley(2))
        search = SearchLoop(problem, CONFIG)
        opt_X, opt_y = search.run()
        self.assertAlmostEqual(opt_y, 0, delta=1e-3)

    def test_simple_dist_rosenbrock_2d(self):
        problem = build_problem("rosenbrock_2d", *build_rosenbrock(2))
        search = SearchLoop(problem, CONFIG)
        opt_X, opt_y = search.run()
        self.assertAlmostEqual(opt_y, 0, delta=1e-3)

if __name__ == '__main__':
    unittest.main()
