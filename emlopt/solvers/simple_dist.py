import logging
import math
import numpy as np

from .base_milp import BaseMILP
from ..emllib.backend import Backend, get_backend
from ..emllib.net.reader.keras_reader import read_keras_probabilistic_sequential
from ..emllib.util import pwl_exp
from ..utils import min_max_scale_in


class SimpleDist(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(SimpleDist, self).__init__(*args, **kwargs)
        self.lambda_ucb = self.cfg['lambda_ucb']

    def solve(self, keras_model, samples_x, samples_y):
        bkd = get_backend(self.cfg['backend'])
        milp_model = bkd.new_model()

        scaled_x_samples = min_max_scale_in(samples_x, np.array(self.problem.input_bounds))

        k_lip = self.compute_klip(samples_x, samples_y)
        current_lambda: float
        if self.lambda_ucb is not None:
            current_lambda = self.lambda_ucb
        else:
            current_lambda = k_lip * ((1-self.current_iteration/self.iterations)**2)

        parsed_mdl = read_keras_probabilistic_sequential(keras_model)
        parsed_mdl, _ = self.propagate_bound(parsed_mdl, timeout=30, timer_logger=self.logger)
        xvars, scaled_xvars, yvars = self.embed_model(bkd, milp_model, parsed_mdl)

        if self.problem.constraint_cb is not None:
            csts = self.problem.constraint_cb(bkd, milp_model, xvars)
            for pc in csts:
                bkd.add_cst(milp_model, *pc)

        stddev = pwl_exp(bkd, milp_model, yvars[1], nnodes=7)

        sample_distance_list = []
        bin_vars = np.empty_like(scaled_x_samples, dtype=object)
        for row in range(scaled_x_samples.shape[0]):
            current_sample_dist = 0
            for feature in range(scaled_x_samples.shape[1]):
                # sum of absolute value
                bin_abs = bkd.var_bin(milp_model, name=f"bin_abs{row}_{feature}")
                bin_vars[row, feature] = bin_abs
                diff = scaled_x_samples[row, feature] - scaled_xvars[feature]
                abs_x = bkd.var_cont(milp_model, lb=0, ub=1, name=f"dist_s_{row}_{feature}")
                M = 10
                bkd.add_cst(milp_model, diff + M*bin_abs >= abs_x)
                bkd.add_cst(milp_model, -diff + M*(1-bin_abs) >= abs_x)
                bkd.add_cst(milp_model, diff <= abs_x)
                bkd.add_cst(milp_model, -diff <= abs_x)
                current_sample_dist += abs_x
            sample_distance_list.append(current_sample_dist)

        # Min distance
        min_dist = bkd.var_cont(milp_model, lb=0, ub=self.problem.input_shape, name="dist")
        for idx, current_sample_dist in enumerate(sample_distance_list):
            bkd.add_cst(milp_model, current_sample_dist >= min_dist, "min dist_"+str(idx))

        # UCB Objective
        ucb = -yvars[0] + \
            current_lambda * stddev + \
            (current_lambda/self.problem.input_shape) * min_dist

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
            "norm_dist": solution['vars']['dist'],
            "mean": solution['vars']['out_mean'],
            "stddev": solution['vars']['exp_out'],
            "logstddev": solution['vars']['out_std'],
            "exp_err": solution['vars']['exp_out'] - math.exp(solution['vars']['out_std']),
            "lambda_ucb": current_lambda
        }
        if self.solution_callback is not None:
            self.solution_callback(main_variables, solution)

        return decision_variables
