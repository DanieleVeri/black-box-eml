import sys, os
sys.path.append('.')

from matplotlib import pyplot as plt
from emlopt.problem import build_problem

def constraint(cplex, xvars):
    x1,x2=xvars
    b = cplex.binary_var()
    M=1000
    return [
        [(x1-2)**2 + (x2+1)**2 - b*M<= 4 , "center_dist"],
        [(x1+2)**2 + (x2-1)**2 - (1-b)*M<= 1 , "center_dist"]
    ]

problem = build_problem("init_points", None, ['int']*2, [[-5,5]]*2, constraint)
x = problem.get_dataset(200, query_obj=False)
plt.figure(figsize=(10,10))
plt.xlim((-5,5))
plt.ylim((-5,5))

fig = plt.gcf()
ax = fig.gca()
ax.add_patch(plt.Circle((2, -1), 2, color='#00ff0033'))
ax.add_patch(plt.Circle((-2, 1), 1, color='#00ff0033'))

plt.scatter(x[:,0], x[:,1])



plt.show()