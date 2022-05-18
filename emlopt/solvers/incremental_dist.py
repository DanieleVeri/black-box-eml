import logging
import math
import numpy as np

from .base_milp import BaseMILP
from  ..emllib.backend import Backend, get_backend
from ..eml import parse_tfp, propagate_bound, embed_model, pwl_exp
from ..utils import min_max_scale_in


class IncrementalDist(BaseMILP):

    def __init__(self, *args, **kwargs):
        super(IncrementalDist, self).__init__(*args, **kwargs)
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

        parsed_mdl = parse_tfp(keras_model)
        parsed_mdl, _ = propagate_bound(bkd, parsed_mdl, self.problem.input_shape, timer_logger=self.logger)

        init_points = 0
        partial_samples_terations = 0
        selection = np.random.choice(list(range(scaled_x_samples.shape[0])), init_points, replace=False)
        selection = scaled_x_samples[selection]

        while True: # Incremental loop
            self.logger.debug(f"Incremental selection: {selection}")
            milp_model = bkd.new_model()

            xvars, scaled_xvars, yvars = embed_model(bkd, milp_model,
                parsed_mdl, self.problem.input_type, self.problem.input_bounds)

            if self.problem.constraint_cb is not None:
                csts = self.problem.constraint_cb(bkd, milp_model, xvars)
                for pc in csts:
                    bkd.add_cst(milp_model, *pc)

            stddev = pwl_exp(bkd, milp_model, yvars[1], nnodes=7)

            if selection.shape[0] > 0:
                sample_distance_list = []
                bin_vars = np.empty_like(selection, dtype=object)
                for row in range(selection.shape[0]):
                    current_sample_dist = 0
                    for feature in range(selection.shape[1]):
                        # sum of absolute value
                        bin_abs = bkd.var_bin(milp_model, name=f"bin_abs{row}_{feature}")
                        bin_vars[row, feature] = bin_abs
                        diff = selection[row, feature] - scaled_xvars[feature]
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
            else:
                min_dist = bkd.var_cont(milp_model, lb=0, ub=0, name="dist")

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

            scaled_decision_variables = self.extract_solution(solution['vars'], scaled=True)
            dists = np.sum(np.abs(scaled_decision_variables - scaled_x_samples), axis=1)
            tot_min_dist, tot_argmin_dist = np.min(dists), np.argmin(dists)
            if selection.shape[0] > 0:
                dists_sel = np.sum(np.abs(scaled_decision_variables - selection), axis=1)
                sel_min_dist = np.min(dists_sel)
            else:
                sel_min_dist = np.inf
            self.logger.debug(f"Tot min dist: {tot_min_dist}, selection min dist: {sel_min_dist}")
            if tot_min_dist < sel_min_dist:
                self.logger.debug(f"Adding closest point to the MILP model")
                partial_samples_terations += 1
                selection = np.concatenate((selection, np.expand_dims(scaled_x_samples[tot_argmin_dist], 0)))
            else:
                self.logger.debug(f"No other closest point found")
                break
        # end of incremental loop

        self.logger.debug(f"Solution: {solution}")
        decision_variables = self.extract_solution(solution['vars'])
        main_variables = {
            "ucb": solution['obj'],
            "norm_dist": solution['vars']['dist'],
            "mean": solution['vars']['out_mean'],
            "stddev": solution['vars']['exp_out'],
            "logstddev": solution['vars']['out_std'],
            "exp_err": solution['vars']['exp_out'] - math.exp(solution['vars']['out_std']),
            "lambda_ucb": current_lambda,
            "partial_iterations": partial_samples_terations
        }
        if self.solution_callback is not None:
            self.solution_callback(main_variables, solution)

        return decision_variables
