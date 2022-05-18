import sys, os
import numpy as np
import unittest
sys.path.append('.')
from matplotlib import pyplot as plt
from emlopt.problem import build_problem
from emlopt.utils import set_seed

CONFIG = {
    "delta": 1e-3,
    "test_mask": [1, 1, 1, 0, 1, 1]
}

def non_convex_constraint(backend, model, xvars):
    x1,x2=xvars
    b = backend.var_bin(model, name="exclusive_bin")
    M=1000
    return [
        [(x1-2)**2 + (x2+1)**2 - b*M<= 4 , "center_dist"],
        [(x1+2)**2 + (x2-1)**2 - (1-b)*M<= 1 , "center_dist"]
    ]

def qadratic_constraint(backend, model, xvars):
    x1,x2=xvars
    return [
        [(x1-2)**2 + (x2+1)**2 <= 4 , "center_dist"],
    ]

def linear_constraint(backend, model, xvars):
    x1,x2=xvars
    return [
        [x1 - x2 <= 0, "ineq"],
    ]

def plot_points(x):
    plt.figure(figsize=(10,10))
    plt.xlim((-5,5))
    plt.ylim((-5,5))
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(plt.Circle((2, -1), 2, color='#00ff0033'))
    ax.add_patch(plt.Circle((-2, 1), 1, color='#00ff0033'))
    plt.scatter(x[:,0], x[:,1])
    plt.show()

class InitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(InitTest, cls).setUpClass()
        set_seed()

    @unittest.skipIf(CONFIG['test_mask'][0]==0, "skip")
    def test_quadratic_integer_cplex(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['int']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=qadratic_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='cplex')
        plot_points(x)
        for p in x:
            self.assertTrue((np.sum(np.square(p-[-2, 1])) <= 1+CONFIG['delta']) or (np.sum(np.square(p-[2, -1])) <= 4+CONFIG['delta']))

    @unittest.skipIf(CONFIG['test_mask'][1]==0, "skip")
    def test_quadratic_real_cplex(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['real']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=qadratic_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='cplex')
        plot_points(x)
        for p in x:
            self.assertTrue((np.sum(np.square(p-[-2, 1])) <= 1+CONFIG['delta']) or (np.sum(np.square(p-[2, -1])) <= 4+CONFIG['delta']))

    @unittest.skipIf(CONFIG['test_mask'][2]==0, "skip")
    def test_non_convex_integer_cplex(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['int']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=non_convex_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='cplex')
        plot_points(x)
        for p in x:
            self.assertTrue((np.sum(np.square(p-[-2, 1])) <= 1+CONFIG['delta']) or (np.sum(np.square(p-[2, -1])) <= 4+CONFIG['delta']))

    @unittest.skipIf(CONFIG['test_mask'][3]==0, "skip")
    def test_non_convex_real_cplex(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['real']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=non_convex_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='cplex')
        plot_points(x)
        for p in x:
            self.assertTrue((np.sum(np.square(p-[-2, 1])) <= 1+CONFIG['delta']) or (np.sum(np.square(p-[2, -1])) <= 4+CONFIG['delta']))

    @unittest.skipIf(CONFIG['test_mask'][4]==0, "skip")
    def test_linear_integer_ortools(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['int']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=linear_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='ortools')
        plot_points(x)
        for p in x:
            self.assertTrue(p[0] <= p[1]+CONFIG['delta'])

    @unittest.skipIf(CONFIG['test_mask'][5]==0, "skip")
    def test_linear_real_ortools(self):
        problem = build_problem(
            name="init_points", fun=None,
            input_type=['real']*2, input_bounds=[[-5,5]]*2,
            constraint_cb=linear_constraint)
        x = problem.get_dataset(200, query_obj=False, backend_type='ortools')
        plot_points(x)
        for p in x:
            self.assertTrue(p[0] <= p[1]+CONFIG['delta'])


if __name__ == '__main__':
    unittest.main()
