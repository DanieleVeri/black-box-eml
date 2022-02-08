import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

def build_probabilistic_regressor(input_shape, depth=4, width=20):
    mdl = tf.keras.Sequential()
    mdl.add(tf.keras.layers.Input(shape=(input_shape,), dtype='float32'))
    for i in range(depth):
        mdl.add(tf.keras.layers.Dense(width, activation='relu'))
    mdl.add(tf.keras.layers.Dense(2, activation='linear'))
    lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], 
                                            scale=tf.keras.backend.exp(t[:, 1:]))
    mdl.add(tfp.layers.DistributionLambda(lf))
    return mdl

def dlambda_likelihood(y_true, dist):
    return -dist.log_prob(y_true)

def plot_prob_predictions(mdl, x, y):
    prob_pred = mdl(np.expand_dims(x, axis=1))
    pred = prob_pred.mean().numpy().ravel()
    std_pred = prob_pred.stddev().numpy().ravel()
    plt.plot(x, y, c="grey")
    plt.plot(x, pred)
    plt.fill_between(x, pred-std_pred, pred+std_pred, 
        alpha=0.3, color='tab:blue', label='+/- std')