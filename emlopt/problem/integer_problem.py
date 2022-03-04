import numpy as np
import docplex.mp.model as cpx
from .base_problem import BaseProblem

class IntegerProblem(BaseProblem):

    def __init__(self, *args, **kwargs):
        super(IntegerProblem, self).__init__(*args, **kwargs)
        self.max_retry = 10

    def get_constrained_dataset(self, n_points, query_obj):
        x = np.zeros((n_points, self.input_shape))              
        num_points = 0
        infeasibilities = 0
        while True:
            cplex = cpx.Model()
            xvars = []
            for i,b in enumerate(self.input_bounds):
                if self.input_type[i] == "int":
                    xvars.append(cplex.integer_var(lb=b[0], ub=b[1], name="x"+str(i)))
                else:
                    xvars.append(cplex.continuous_var(lb=b[0], ub=b[1], name="x"+str(i)))
            
            csts = self.constraint_cb(cplex, xvars)
            # create restricted problem
            restriction = 0 if num_points < n_points//2 else np.random.uniform()
            for pc in csts:
                if not pc[0].right_expr.equals(0):
                    if pc[0].sense.value == 1: # <=
                        pc[0].right_expr -= restriction*pc[0].right_expr
                    elif pc[0].sense.value == 3: # >=
                        pc[0].right_expr += restriction*pc[0].right_expr
                cplex.add_constraint(*pc)

            # linear random objective
            obj = 0
            for i, var in enumerate(xvars):
                obj += var * (np.random.uniform()*2-1)
            cplex.set_objective("min", obj)
            
            # solve
            cplex.set_time_limit(30)
            sol = cplex.solve()
            if sol is None:
                infeasibilities+=1
                continue
            for i in range(self.input_shape):
                x[num_points, i] = sol["x"+str(i)]

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
