import math
import numpy as np

def mccormick(x):
    x1,x2 = x[0],x[1]
    return math.sin(x1+x2) + math.pow(x1-x2,2)-1.5*x1+2.5*x2+1

def polynomial(x):
    return 0.5*(0.1*math.pow(5*x-1, 4) - 0.4*math.pow(5*x-1, 3) + 0.5*(5*x-1))

def polynomial_stochastic(x):
    rnd = (np.random.random()-0.5) * np.abs(np.sin(x*5))/2
    return rnd+0.5*(0.1*math.pow(5*x-1, 4) - 0.4*math.pow(5*x-1, 3) + 0.5*(5*x-1))

def build_rosenbrock(rosenbrock_dim):
    def f(x):
        y=0
        for i in range(rosenbrock_dim-1):
            y+= 100 * math.pow(x[i+1] - x[i]*x[i], 2) + math.pow(1-x[i], 2)
        return y
    vtypes = ["real"]*rosenbrock_dim
    bounds = [[-2, 2]]*rosenbrock_dim
    return f, vtypes, bounds