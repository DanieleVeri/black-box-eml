import numpy as np
from ..emllib.backend import get_backend
from .base_problem import BaseProblem

class IntegerProblem(BaseProblem):

    def __init__(self, *args, **kwargs):
        super(IntegerProblem, self).__init__(*args, **kwargs)
        self.max_retry = 10

    def get_constrained_dataset(self, n_points, query_obj, backend_type):
        backend = get_backend(backend_type)
        x = np.zeros((n_points, self.input_shape))
        num_points = 0
        infeasibilities = 0
        while True:
            model = backend.new_model()
            xvars = []
            for i,b in enumerate(self.input_bounds):
                if self.input_type[i] == "int":
                    xvars.append(backend.var_int(model, lb=b[0], ub=b[1], name="x"+str(i)))
                else:
                    xvars.append(backend.var_cont(model, lb=b[0], ub=b[1], name="x"+str(i)))

            csts = self.constraint_cb(backend, model, xvars)
            for pc in csts:
                backend.add_cst(model, *pc)
            ## create restricted integer problem
            # restriction = 0 if num_points < n_points//2 else np.random.uniform()
            # for pc in csts:
            #     if backend_type != 'cplex':
            #         backend.add_cst(model, *pc)
            #         continue
            #     if not pc[0].right_expr.equals(0):
            #         if pc[0].sense.value == 1: # <=
            #             pc[0].right_expr -= restriction*pc[0].right_expr
            #         elif pc[0].sense.value == 3: # >=
            #             pc[0].right_expr += restriction*pc[0].right_expr
            #     backend.add_cst(model, *pc)

            # linear random objective
            obj = 0
            for i, var in enumerate(xvars):
                obj += var * (np.random.uniform()*2-1)
            backend.set_obj(model, "min", obj)

            # solve
            solution = backend.solve(model, 30)
            if solution['status'] == 'infeasible':
                infeasibilities+=1
                continue
            for i in range(self.input_shape):
                x[num_points, i] = solution['vars']["x"+str(i)]

            num_points += 1
            if num_points == n_points: break
            if infeasibilities == n_points*self.max_retry:
                raise Exception("Number of infeasibilities excedeed.")

        if not query_obj:
            return x

        # eval fun
        y = np.zeros((n_points))
        for i in range(n_points):
            y[i] = self.fun(x[i, :])

        return x,y
