import math

def objective(x):
    x1,x2 = x[0],x[1]
    return math.sin(x1+x2) + math.pow(x1-x2,2)-1.5*x1+2.5*x2+1