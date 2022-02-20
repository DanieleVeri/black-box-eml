import logging
import math
import numpy as np
import docplex.mp.model as cpx
from eml.backend import cplex_backend
from .base_milp import BaseMILP
from ..eml import parse_tfp, propagate_bound, embed_model, pwl_exp
from ..utils import min_max_scale_in


class IncrementalDist(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(IncrementalDist, self).__init__(*args, **kwargs)
        self.lambda_ucb = self.cfg['lambda_ucb']

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

        INIT_POINTS=0
        partial_samples_terations = 0
        selection = np.random.choice(list(range(scaled_x_samples.shape[0])), INIT_POINTS, replace=False)
        selection = scaled_x_samples[selection]
        while True:
            self.logger.debug(f"Incremental selection: {selection}")
            cplex_model = cpx.Model()

            xvars, scaled_xvars, yvars = embed_model(bkd, cplex_model, 
                parsed_mdl, self.problem.input_type, self.problem.input_bounds)

            if self.problem.constraint_cb is not None:
                csts = self.problem.constraint_cb(xvars)
                for pc in csts:
                    cplex_model.add_constraint(*pc)

            stddev = pwl_exp(bkd, cplex_model, yvars[1], nnodes=7)

            if selection.shape[0] > 0:
                sample_distance_list = []
                bin_vars = np.empty_like(selection, dtype=object)
                for row in range(selection.shape[0]):
                    current_sample_dist = 0
                    for feature in range(selection.shape[1]):
                        # sum of absolute value
                        bin_abs = cplex_model.binary_var(name=f"bin_abs{row}_{feature}")
                        bin_vars[row, feature] = bin_abs
                        diff = selection[row, feature] - scaled_xvars[feature]
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
            else:
                min_dist = cplex_model.continuous_var(lb=0, ub=0, name="dist")

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

            opt_x = self.extract_solution(solution, scaled=True)
            dists = np.sum(np.abs(opt_x - scaled_x_samples), axis=1)
            tot_min_dist, tot_argmin_dist = np.min(dists), np.argmin(dists)
            if selection.shape[0] > 0:
                dists_sel = np.sum(np.abs(opt_x - selection), axis=1)
                sel_min_dist = np.min(dists_sel)
            else:
                sel_min_dist = np.inf
            self.logger.debug(f"Tot min dist: {tot_min_dist}, selection min dist: {sel_min_dist}")
            if tot_min_dist < sel_min_dist:
                partial_samples_terations += 1
                selection = np.concatenate((selection, np.expand_dims(scaled_x_samples[tot_argmin_dist], 0)))
            else:
                break

            solution_log = {
                "ucb": solution.objective_value,
                "norm_dist": solution['dist'],
                "mean": solution['out_mean'],
                "stddev": solution['exp_out'],
                "exp_err": solution['exp_out'] - math.exp(solution['out_std']),
                "lambda_ucb": current_lambda,
                "partial_iterations": partial_samples_terations
            }
            self.logger.debug(f"MILP solution:\n{solution_log}")
            if self.logger.level == logging.DEBUG:
                solution.solve_details.print_information()

        if self.solution_callback is not None:
            self.solution_callback(solution_log)

        return self.extract_solution(solution)
        