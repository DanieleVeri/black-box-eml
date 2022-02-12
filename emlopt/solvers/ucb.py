import logging
import math
import numpy as np
import docplex.mp.model as cpx
from eml.backend import cplex_backend
from .base_milp import BaseMILP
from ..eml import parse_tfp, propagate_bound, embed_model, pwl_exp


class UCB(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(UCB, self).__init__(*args, **kwargs)
        self.lambda_ucb = self.cfg['lambda_ucb']

    def solve(self, keras_model, samples_x, samples_y):
        cplex_model = cpx.Model()
        bkd = cplex_backend.CplexBackend()

        parsed_mdl = parse_tfp(keras_model)
        parsed_mdl, _ = propagate_bound(bkd, parsed_mdl, self.problem.input_shape, timer_logger=self.logger)
        xvars, scaled_xvars, yvars = embed_model(bkd, cplex_model, 
            parsed_mdl, self.problem.input_type, self.problem.input_bounds)

        if self.problem.constraint_cb is not None:
            csts = self.problem.constraint_cb(xvars)
            for pc in csts:
                cplex_model.add_constraint(*pc)

        stddev = pwl_exp(bkd, cplex_model, yvars[1], nnodes=7)

        # UCB Objective
        ucb = -yvars[0] + self.lambda_ucb * stddev 
            
        cplex_model.set_objective('max', ucb)
        cplex_model.set_time_limit(self.solver_timeout)
        self.cplex_deterministic(cplex_model)
        if self.logger.level == logging.DEBUG:
            self.cplex_extensive_log(cplex_model)
        solution = cplex_model.solve()

        if solution is None:
            raise Exception("Not feasible")

        solution_log = {
            "ucb": solution.objective_value,
            "mean": solution['out_mean'],
            "stddev": solution['exp_out'],
            "exp_err": solution['exp_out'] - math.exp(solution['out_std']),
            "lambda_ucb": self.lambda_ucb
        }
        self.logger.debug(f"MILP solution:\n{solution_log}")
        if self.logger.level == logging.DEBUG:
            solution.solve_details.print_information()

        if self.solution_callback is not None:
            self.solution_callback(solution_log)

        return self.extract_solution(solution)
        