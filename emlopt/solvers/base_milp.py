import sys
import numpy as np
from ..problem import BaseProblem
from ..utils import timer


class BaseMILP:

    def __init__(self, problem, milp_cfg, iterations, logger):
        self.problem: BaseProblem = problem
        self.cfg = milp_cfg
        self.logger = logger
        self.solver_timeout: int = milp_cfg['solver_timeout']
        self.iterations = iterations
        self.current_iteration = 0
        self.not_improve_iteration = 0
        self.solution_callback = None

    def solve(self, keras_model, samples_x, sample_y):
        raise NotImplementedError

    @timer
    def optimize_acquisition_function(self, keras_model, samples_x, sample_y):
        self.logger.info(f"{self.__class__.__name__} solver:")
        return self.solve(keras_model, samples_x, sample_y)

    def compute_klip(self, x, y):
        k_lip = 0
        qlist = []
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                delta_y = np.abs(y[i]-y[j])
                delta_x = np.sum(np.abs(x[i]-x[j]))/self.problem.input_shape
                if delta_x >= 1e-6: #eps to avoid inf
                    k_lip = max(k_lip, delta_y/delta_x)
                    qlist.append(delta_y/delta_x)
        self.logger.debug(f"K-Lipschitz max: {k_lip} - 95° percentile: {np.percentile(qlist, 95, interpolation='linear')}")
        return k_lip

    def extract_solution(self, solution_vars, scaled=False):
        opt_x = np.zeros(self.problem.input_shape)
        for i in range(self.problem.input_shape):
            if scaled:
                opt_x[i] = solution_vars["norm_x"+str(i)]
            else:
                opt_x[i] = solution_vars["x"+str(i)]
        return opt_x
