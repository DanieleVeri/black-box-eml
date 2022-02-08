import pandas as pd
import numpy as np
import docplex.mp.model as cpx
from .base_problem import BaseProblem

class ConvexRealProblem(BaseProblem):

    def __init__(self, *args, **kwargs):
        super(ConvexRealProblem, self).__init__(*args, **kwargs)

    def get_constrained_dataset(self, n_points):
        x = np.zeros((n_points, self.input_shape))
        bounds = []
        for i, b in enumerate(self.input_bounds):
            if self.input_type[i] == "int":
                bounds.append((int(b[0]), int(b[1])))
            else:
                bounds.append((float(b[0]), float(b[1])))

        n_points_boundaries = n_points // 2
        for p in range(n_points_boundaries):
            cplex = cpx.Model()
            xvars = []
            for i,b in enumerate(self.input_bounds):
                if self.input_type[i] == "int":
                    xvars.append(cplex.integer_var(lb=b[0], ub=b[1], name="x"+str(i)))
                else:
                    xvars.append(cplex.continuous_var(lb=b[0], ub=b[1], name="x"+str(i)))
            
            csts = self.constraint_cb(xvars)
            for pc in csts:
                cplex.add_constraint(*pc)

            # linear random objective (boundaries)
            obj = 0
            for var in xvars:
                obj += var * (np.random.uniform()*2-1)
            cplex.set_objective("max", obj)
            
            # solve
            cplex.set_time_limit(30)
            sol = cplex.solve()
            for i in range(self.input_shape):
                x[p, i] = sol["x"+str(i)]

        # Interpolation  
        for p in range(n_points_boundaries, n_points):
            pidx = np.random.choice(n_points_boundaries, 2, replace=False)
            p0, p1 = x[pidx[0]], x[pidx[1]]
            step = np.random.uniform()
            df = pd.DataFrame([p0, [np.nan]*self.input_shape, p1], 
                              index=[0, step, 1])
            df = df.interpolate(method="index")
            x[p] = df.iloc[1].values

        # eval fun
        y = np.zeros((n_points))
        for i in range(n_points):
            y[i] = self.fun(x[i, :])

        return x,y