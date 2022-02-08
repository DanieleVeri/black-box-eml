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
        cplex_model = cpx.Model()
        bkd = cplex_backend.CplexBackend()

        scaled_x_samples = min_max_scale_in(samples_x, np.array(self.problem.input_bounds))

        k_lip = self.compute_klip(samples_x, samples_y)
        
        current_lambda: float
        if self.lambda_ucb is not None:
            current_lambda = self.lambda_ucb
        else:
            current_lambda = k_lip * ((1-self.current_iteration/self.iterations)**2)

        parsed_mdl = parse_tfp(keras_model)
        parsed_mdl, _ = propagate_bound(bkd, parsed_mdl, self.problem.input_shape)
        xvars, scaled_xvars, yvars = embed_model(bkd, cplex_model, 
            parsed_mdl, self.problem.input_type, self.problem.input_bounds)

        if self.problem.constraint_cb is not None:
            csts = self.problem.constraint_cb(xvars)
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

        # > successors
        # for feature in range(bin_vars.shape[1]):
        #     order = np.argsort(scaled_x_samples[:, feature])
        #     for i in range(len(order)):
        #         for j in range(i, len(order)):
        #             cplex_model.add_constraint(bin_vars[order[i], feature] >= bin_vars[order[j], feature])
       
        # max 1 switch
        # for feature in range(bin_vars.shape[1]):
        #     order = np.argsort(scaled_x_samples[:, feature])
        #     summa = (1 >= bin_vars[order[0], feature]+1)
        #     for i in range(len(order)-1):
        #         summa += (bin_vars[order[i], feature] >= bin_vars[order[i+1], feature]+1)
        #     summa += (bin_vars[order[-1], feature] >= 1)
        #     cplex_model.add_constraint(summa == 1)
            
        # triangular ineq next only
        # for id_xy in range(len(sample_distance_list)-1):
        #     s_xy = sample_distance_list[id_xy]
        #     id_xz = id_xy+1
        #     s_xz = sample_distance_list[id_xz]
        #     sample_distance = np.sum(np.abs(scaled_x_samples[id_xy] - scaled_x_samples[id_xz]))
        #     cplex_model.add_constraint(s_xy <= s_xz + sample_distance, "traingular inequality")

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

        solution_log = {
            "ucb": solution.objective_value,
            "norm_dist": solution['dist'],
            "mean": solution['out_mean'],
            "stddev": solution['exp_out'],
            "exp_err": solution['exp_out'] - math.exp(solution['out_std']),
            "lambda_ucb": current_lambda
        }
        self.logger.debug(f"MILP solution:\n{solution_log}")
        if self.logger.level == logging.DEBUG:
            solution.solve_details.print_information()

        if self.solution_callback is not None:
            self.solution_callback(solution_log)

        return self.extract_solution(solution)
        