import math

def build_ackley(dim):
    def f(x):
        v1=0
        for i in range(dim):
            v1 += x[i]*x[i]
        v1 = math.exp(-0.2*math.sqrt(v1/dim))
        v2=0
        for i in range(dim):
            v2 += math.cos(2*math.pi*x[i])
        v2 = math.exp(v2/dim)
        return -20*v1-v2+20+math.e

    vtypes = ["real"]*dim
    bounds = [[-5, 10]]*dim
    return f, vtypes, bounds

def constraint_scbo(cplex, xvars):
    r = 5
    acc1 = 0
    for xi in xvars:
        acc1 += xi*xi
    acc2 = 0
    for xi in xvars:
        acc2 += xi
    return [[acc1 <= r*r, "center_dist"], [acc2 <= 0, "cst2"]]