import math
import numpy as np
from scipy.stats import norm
import tensorflow as tf
from eml.net import embed
from eml.net.reader import keras_reader
from eml.net.process import fwd_bound_tighthening
from eml.util import encode_pwl
from .utils import timer
from .problem import BaseProblem


def parse_tfp(model):
    in_shape = model.input_shape[1]
    mdl_no_dist = tf.keras.Sequential()
    mdl_no_dist.add(tf.keras.layers.Input(shape=(in_shape,), dtype='float32'))
    for i in range(len(model.layers)-2):
        w = model.layers[i].weights[1].shape[0]
        mdl_no_dist.add(tf.keras.layers.Dense(w, activation='relu'))
    mdl_no_dist.add(tf.keras.layers.Dense(2, activation='linear'))

    mdl_no_dist.set_weights(model.get_weights())

    nn = keras_reader.read_keras_sequential(mdl_no_dist)
    return nn

@timer
def propagate_bound(bkd, parsed_model, shape):
    bounds = np.array([[0,1]]*shape)
    parsed_model.layer(0).update_lb(bounds[:,0])
    parsed_model.layer(0).update_ub(bounds[:,1])
    fwd_bound_tighthening(bkd, parsed_model, timelimit=30)
    return parsed_model

def embed_model(bkd, cplex, parsed_model, vtype, bounds):
    mean_lb = parsed_model.layer(-1).lb()[0]  # bounds computed with propagate bounds method
    mean_ub = parsed_model.layer(-1).ub()[0]
    std_lb = parsed_model.layer(-1).lb()[1]
    std_ub = parsed_model.layer(-1).ub()[1]

    xvars = []
    norm_xvars = []
    for i,b in enumerate(bounds):
        if vtype[i] == "int":
            xvars.append(cplex.integer_var(lb=b[0], ub=b[1], name="x"+str(i)))
        else:
            xvars.append(cplex.continuous_var(lb=b[0], ub=b[1], name="x"+str(i)))
        # NN scaled input
        norm_xvars.append(cplex.continuous_var(lb=0, ub=1, name="norm_x"+str(i)))
        cplex.add_constraint(norm_xvars[-1] * (b[1] - b[0]) == xvars[-1] - b[0])

    yvars = [cplex.continuous_var(lb=mean_lb, ub=mean_ub, name="out_mean"), 
        cplex.continuous_var(lb=std_lb, ub=std_ub, name="out_std")]

    embed.encode(bkd, parsed_model, cplex, norm_xvars, yvars, 'nn')
    return xvars, norm_xvars, yvars

def pwl_exp(bkd, cplex, var, nnodes=7):
    xx = np.linspace(var.lb, var.ub, nnodes)
    yy = np.array(list(map(math.exp, xx)))
    v = [cplex.continuous_var(lb=0, ub=np.max(yy), name="exp_out"), 
         var]
    encode_pwl(bkd, cplex, v, [yy,xx])
    return v[0]

def pwl_abs(bkd, cplex, var):
    xx = np.array([-1, 0, 1])
    yy = np.array([1, 0, 1])
    v = [cplex.continuous_var(lb=0, ub=1), 
         var]
    encode_pwl(bkd, cplex, v, [yy,xx])
    return v[0]

def pwl_normal_cdf(bkd, cplex, var, nnodes=11):
    xx = np.linspace(var.lb, var.ub, nnodes)
    yy = np.array(list(map(norm.cdf, xx)))
    v = [cplex.continuous_var(lb=0, ub=1, name="ncdf_out"), 
         var]
    encode_pwl(bkd, cplex, v, [yy,xx])
    return v[0]

def pwl_normal_pdf(bkd, cplex, var, nnodes=11):
    xx = np.linspace(var.lb, var.ub, nnodes)
    yy = np.array(list(map(norm.pdf, xx)))
    v = [cplex.continuous_var(lb=0, ub=1, name="npdf_out"), 
         var]
    encode_pwl(bkd, cplex, v, [yy,xx])
    return v[0]

def pwl_sample_dist(bkd, cplex, vars, samples, vtype, bounds, nnodes=20):
    mapf = lambda x: np.min(np.sum(np.abs(samples - x), axis=1)) / len(bounds)
    p = BaseProblem(None, mapf, vtype, bounds)
    x,y = p.get_grid(nnodes)
    ub = np.diff(np.array(bounds), axis=1)
    ub = np.sum(ub)
    v = [cplex.continuous_var(lb=0, ub=ub, name="dist_out")]+vars
    encode_pwl(bkd, cplex, v, [y,*x.T])
    return v[0]