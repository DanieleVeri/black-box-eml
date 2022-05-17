import sys, os
import unittest
sys.path.append('.')
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
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

    def test_simple_dist_polynomial_1d(self):
        problem = build_problem("polynomial_1d", polynomial, ['real'], [[0,1]])
        search = SearchLoop(problem, CONFIG)
        opt_X, opt_y = search.run()
        self.assertAlmostEqual(opt_y, 0.619, delta=1e-3)

# NAME="mccormick_2d"
# problem = build_problem(NAME, mccormick, ['real', 'real'], [[-1.5, 4], [-3, 4]])
# search = SearchLoop(problem, CONFIG)
# wandb_cfg['experiment_name'] = NAME
# with WandbContext(wandb_cfg, search): search.run()
#
# NAME="ackley_2d"
# problem = build_problem(NAME, *build_ackley(2))
# search = SearchLoop(problem, CONFIG)
# wandb_cfg['experiment_name'] = NAME
# with WandbContext(wandb_cfg, search): search.run()
#
# NAME="rosenbrock2d"
# problem = build_problem(NAME, *build_rosenbrock(2))
# search = SearchLoop(problem, CONFIG)
# wandb_cfg['experiment_name'] = NAME
# with WandbContext(wandb_cfg, search): search.run()

if __name__ == '__main__':
    unittest.main()
