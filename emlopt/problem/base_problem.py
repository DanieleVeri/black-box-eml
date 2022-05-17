import numpy as np


class BaseProblem:

    def __init__(self, name, fun, input_type, input_bounds, constraint_cb=None, stocasthic=False, backend=None):
        self.name = name
        self.fun = fun
        self.input_type = input_type
        self.input_bounds = input_bounds
        self.input_shape = len(self.input_bounds)
        self.constraint_cb = constraint_cb
        self.stocasthic = stocasthic
        self.backend = backend

    def get_constrained_dataset(self, n_points):
        raise NotImplementedError

    def get_dataset(self, n_points, query_obj=True):
        if self.constraint_cb is None:
            return self.get_unconstrained_dataset(n_points, query_obj)
        else:
            if self.backend is None:
                raise Exception("Backend required for constrained dataset.")
            return self.get_constrained_dataset(n_points, query_obj)

    def get_grid(self, n_points, query_obj=True):
        x_list = []
        for i, b in enumerate(self.input_bounds):
            lb = b[0]
            ub = b[1]
            if self.input_type[i] == "int":
                x_list.append(np.arange(lb, ub, max(1, (ub-lb)//n_points)))
            else:
                x_list.append(np.arange(lb, ub, (ub-lb)/n_points))
        x = np.array(np.meshgrid(*x_list)).reshape(self.input_shape,-1).T
        if not query_obj:
            return x
        y = np.zeros((x.shape[0]))
        for i in range(x.shape[0]):
            y[i] = self.fun(x[i, :])
        return x, y

    def get_unconstrained_dataset(self, n_points, query_obj):
        x = np.random.rand(n_points, self.input_shape)
        for i, b in enumerate(self.input_bounds):
            lb = b[0]
            ub = b[1]
            if self.input_type[i] == "int":
                x[:,i] = np.random.randint(lb, high=ub, size=n_points)
            else:
                x[:,i] *= ub - lb
                x[:,i] += lb
        if not query_obj:
            return x
        y = np.zeros((n_points))
        for i in range(n_points):
            y[i] = self.fun(x[i, :])
        return x, y
