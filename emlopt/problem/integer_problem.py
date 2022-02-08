import numpy as np
import docplex.mp.model as cpx
from skopt.sampler import Lhs
from skopt.space import Space
from .base_problem import BaseProblem

class IntegerProblem(BaseProblem):

    def __init__(self, *args, **kwargs):
        super(IntegerProblem, self).__init__(*args, **kwargs)

    def get_dataset(self, n_points):
        x = np.zeros((n_points, self.input_shape))
        bounds = []
        for i, b in enumerate(self.input_bounds):
            if self.input_type[i] == "int":
                bounds.append((int(b[0]), int(b[1])))
            else:
                bounds.append((float(b[0]), float(b[1])))
                
        space = Space(bounds)
        lhs = Lhs(lhs_type="classic", criterion=None, iterations=1000)
        lhs_samples = lhs.generate(space.dimensions, n_points)

        p = 0
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
            for pc in csts:
                if pc[0].sense.value == 1: # <=
                    pc[0].right_expr -= np.random.uniform()*pc[0].right_expr
                elif pc[0].sense.value == 3: # >=
                    pc[0].right_expr += np.random.uniform()*pc[0].right_expr
                cplex.add_constraint(*pc)

            ## quadratic random objective (boundaries)
            obj = 0
            for i, var in enumerate(xvars):
                obj += (var-lhs_samples[p][i]) ** 2
            cplex.set_objective("min", obj)
            
            # solve
            cplex.set_time_limit(30)
            sol = cplex.solve()
            if sol is None:
                print("infeasible")
                continue
            for i in range(self.input_shape):
                x[p, i] = sol["x"+str(i)]

            p += 1
            if p == n_points: break

        # eval fun
        y = np.zeros((n_points))
        for i in range(n_points):
            y[i] = self.fun(x[i, :])

        return x,y
