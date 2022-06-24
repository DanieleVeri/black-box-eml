import sys
import numpy as np
from ..problem import BaseProblem
from ..utils import timer
from ..emllib.net.process import fwd_bound_tighthening, ibr_bounds
from ..emllib.backend import get_backend
from ..emllib.net import embed


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
        self.logger.debug(f"{self.__class__.__name__} solver:")
        return self.solve(keras_model, samples_x, sample_y)

    @timer
    def propagate_bound(self, parsed_model, timeout=30):
        backend = get_backend(self.cfg['backend'])
        bounds = np.array([[0,1]]*self.problem.input_shape)
        parsed_model.layer(0).update_lb(bounds[:,0])
        parsed_model.layer(0).update_ub(bounds[:,1])
        method = self.cfg.get('bound_propagation', 'both') # both is default
        if method == 'ibr':
            self.logger.debug(f"Using Interval Based Reasoning bound propagation")
            ibr_bounds(parsed_model)
        elif method == 'milp':
            self.logger.debug(f"Using MILP forward bound tighthening bound propagation")
            fwd_bound_tighthening(backend, parsed_model, timelimit=timeout)
        elif method == 'both':
            self.logger.debug(f"Using IBR and then MILP bound propagation")
            ibr_bounds(parsed_model)
            fwd_bound_tighthening(backend, parsed_model, timelimit=timeout)
        elif method == 'domain':
            self.logger.debug(f"Using IBR and then MILP bound propagation with domain constraints")

            # link normalized vars to the decision variables which are used in the constraints
            def wrap_constraint_cb(bkd, mdl, norm_xvars):
                xvars = []
                for i,b in enumerate(self.problem.input_bounds):
                    if self.problem.input_type[i] == "int":
                        xvars.append(bkd.var_int(mdl, lb=b[0], ub=b[1], name="x"+str(i)))
                    else:
                        xvars.append(bkd.var_cont(mdl, lb=b[0], ub=b[1], name="x"+str(i)))
                    bkd.cst_eq(mdl, norm_xvars[i] * (b[1] - b[0]), xvars[-1] - b[0], "cst_norm_x"+str(i))
                return self.problem.constraint_cb(bkd, mdl, xvars)

            ibr_bounds(parsed_model)
            fwd_bound_tighthening(backend, parsed_model, timelimit=timeout, constraint_cb=wrap_constraint_cb)
        else:
            raise Exception('Invalid bound propagation method')
        return parsed_model

    def embed_model(self, bkd, milp_model, parsed_model, new_bounds=None):
        mean_lb = parsed_model.layer(-1).lb()[0]  # bounds computed with propagate bounds method
        mean_ub = parsed_model.layer(-1).ub()[0]
        std_lb = parsed_model.layer(-1).lb()[1]
        std_ub = parsed_model.layer(-1).ub()[1]
        xvars = []
        norm_xvars = []
        bounds = self.problem.input_bounds if new_bounds is None else new_bounds
        for i,b in enumerate(bounds):
            if self.problem.input_type[i] == "int":
                xvars.append(bkd.var_int(milp_model, lb=b[0], ub=b[1], name="x"+str(i)))
            else:
                xvars.append(bkd.var_cont(milp_model, lb=b[0], ub=b[1], name="x"+str(i)))
            # NN scaled input
            norm_xvars.append(bkd.var_cont(milp_model, lb=0, ub=1, name="norm_x"+str(i)))
            bkd.cst_eq(milp_model, norm_xvars[-1] * (b[1] - b[0]), xvars[-1] - b[0], "cst_norm_x"+str(i))

        yvars = [bkd.var_cont(milp_model, lb=mean_lb, ub=mean_ub, name="out_mean"),
            bkd.var_cont(milp_model, lb=std_lb, ub=std_ub, name="out_std")]

        embed.encode(bkd, parsed_model, milp_model, norm_xvars, yvars, 'nn')
        return xvars, norm_xvars, yvars

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
