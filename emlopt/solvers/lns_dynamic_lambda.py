from copy import deepcopy
import logging
import math
import numpy as np
import docplex.mp.model as cpx
from eml.backend import cplex_backend
from .base_milp import BaseMILP
from ..eml import parse_tfp, propagate_bound, embed_model, pwl_exp
from ..utils import min_max_scale_in


class LNSDynamicLambdaDist(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(LNSDynamicLambdaDist, self).__init__(*args, **kwargs)
        self.lambda_ucb = self.cfg['lambda_ucb']
        self.sub_problems = self.cfg['sub_problems']

    def solve(self, keras_model, samples_x, samples_y):
        bkd = cplex_backend.CplexBackend()
        scaled_x_samples = min_max_scale_in(samples_x, np.array(self.problem.input_bounds))
        k_lip = self.compute_klip(samples_x, samples_y)
        
        current_lambda: float
        if self.lambda_ucb is not None:
            current_lambda = self.lambda_ucb
        else:
            current_lambda = k_lip * ((1-self.current_iteration/self.iterations)**2)

        parsed_mdl = parse_tfp(keras_model)
        parsed_mdl, _ = propagate_bound(bkd, parsed_mdl, self.problem.input_shape, timer_logger=self.logger)

        final_solution = None
        feature_fixed = self.problem.input_shape - (self.problem.input_shape // self.sub_problems)
        for lns_i in range(self.sub_problems):
            self.logger.debug(f"Fixed feature selection: {feature_fixed}")
            cplex_model = cpx.Model()

            best = scaled_x_samples[np.argmin(samples_y)]
            selection = np.random.choice(self.problem.input_shape, feature_fixed, replace=False)
            self.logger.debug(f"LNS Random selection {selection} best: {best}")
            pbounds = deepcopy(self.problem.input_bounds)
            for v in selection:
                pbounds[v] = [best[v], best[v]]

            xvars, scaled_xvars, yvars = embed_model(bkd, cplex_model, 
                parsed_mdl, self.problem.input_type, pbounds)

            if self.problem.constraint_cb is not None:
                csts = self.problem.constraint_cb(cplex_model, xvars)
                for pc in csts:
                    cplex_model.add_constraint(*pc)

            stddev = pwl_exp(bkd, cplex_model, yvars[1], nnodes=7)

            sample_distance_list = []
            bin_vars = np.empty_like(scaled_x_samples, dtype=object)
            for row in range(scaled_x_samples.shape[0]):
                current_sample_dist = 0
                for feature in range(scaled_x_samples.shape[1]):
                    # sum of absolute value
                    bin_abs = cplex_model.binary_var(name=f"bin_abs{row}_{feature}")
                    bin_vars[row, feature] = bin_abs
                    diff = scaled_x_samples[row, feature] - scaled_xvars[feature]
                    abs_x = cplex_model.continuous_var(lb=0, ub=1)
                    M = 10
                    cplex_model.add_constraint(diff + M*bin_abs >= abs_x)
                    cplex_model.add_constraint(-diff + M*(1-bin_abs) >= abs_x)
                    cplex_model.add_constraint(diff <= abs_x)
                    cplex_model.add_constraint(-diff <= abs_x)
                    current_sample_dist += abs_x
                sample_distance_list.append(current_sample_dist)

            # Min distance
            min_dist = cplex_model.continuous_var(lb=0, ub=self.problem.input_shape, name="dist")
            for current_sample_dist in sample_distance_list:
                cplex_model.add_constraint(current_sample_dist >= min_dist, "min dist")

            # UCB Objective
            ucb = -yvars[0] + \
                current_lambda * stddev + \
                (current_lambda/self.problem.input_shape) * min_dist

            cplex_model.set_objective('max', ucb)
            cplex_model.set_time_limit(self.solver_timeout)
            self.cplex_deterministic(cplex_model)
            if self.logger.level == logging.DEBUG:
                self.cplex_extensive_log(cplex_model)
            solution = cplex_model.solve()

            if solution is None:
                raise Exception("Not feasible")

            if final_solution is not None:
                if solution.objective_value > final_solution.objective_value:
                    final_solution = solution
            else:
                final_solution = solution
            
        solution_log = {
            "ucb": final_solution.objective_value,
            "norm_dist": final_solution['dist'],
            "mean": final_solution['out_mean'],
            "stddev": final_solution['exp_out'],
            "exp_err": final_solution['exp_out'] - math.exp(final_solution['out_std']),
            "lambda_ucb": current_lambda,
        }
        self.logger.debug(f"MILP solution:\n{solution_log}")
        if self.logger.level == logging.DEBUG:
            final_solution.solve_details.print_information()

        if self.solution_callback is not None:
            self.solution_callback(solution_log)

        return self.extract_solution(final_solution)
        