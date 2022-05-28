import logging
import math
from .base_milp import BaseMILP
from ..emllib.backend import Backend, get_backend
from ..emllib.net.reader.keras_reader import read_keras_probabilistic_sequential
from ..emllib.util import pwl_exp


class UCB(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(UCB, self).__init__(*args, **kwargs)
        self.lambda_ucb = self.cfg['lambda_ucb']

    def solve(self, keras_model, samples_x, samples_y):
        bkd = get_backend(self.cfg['backend'])
        milp_model = bkd.new_model()

        parsed_mdl = read_keras_probabilistic_sequential(keras_model)
        parsed_mdl, _ = self.propagate_bound(parsed_mdl, timeout=30, timer_logger=self.logger)
        xvars, scaled_xvars, yvars = self.embed_model(bkd, milp_model, parsed_mdl)

        if self.problem.constraint_cb is not None:
            csts = self.problem.constraint_cb(bkd, milp_model, xvars)
            for pc in csts:
                bkd.add_cst(milp_model, *pc)

        stddev = pwl_exp(bkd, milp_model, yvars[1], nnodes=7)

        # UCB Objective
        ucb = -yvars[0] + self.lambda_ucb * stddev

        bkd.set_obj(milp_model, 'max', ucb)
        bkd.set_determinism(milp_model)
        if self.logger.level == logging.DEBUG:
            bkd.set_extensive_log(milp_model)
        solution = bkd.solve(milp_model, self.solver_timeout)

        if solution['status'] == 'infeasible':
            raise Exception("Not feasible")

        self.logger.debug(f"Solution: {solution}")
        decision_variables = self.extract_solution(solution['vars'])
        main_variables = {
            "ucb": solution['obj'],
            "mean": solution['vars']['out_mean'],
            "stddev": solution['vars']['exp_out'],
            "logstddev": solution['vars']['out_std'],
            "exp_err": solution['vars']['exp_out'] - math.exp(solution['vars']['out_std']),
            "lambda_ucb": self.lambda_ucb
        }
        if self.solution_callback is not None:
            self.solution_callback(main_variables, solution)

        return decision_variables
