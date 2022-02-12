import tensorflow as tf
import tensorflow_probability as tfp

def build_probabilistic_regressor(input_shape, depth=4, width=20):
    mdl = tf.keras.Sequential()
    mdl.add(tf.keras.layers.Input(shape=(input_shape,), dtype='float32'))
    for i in range(depth):
        mdl.add(tf.keras.layers.Dense(width, activation='relu'))
    mdl.add(tf.keras.layers.Dense(2, activation='linear'))
    lf = lambda t: tfp.distributions.Normal(loc=t[:, :1], scale=tf.keras.backend.exp(t[:, 1:]))
    mdl.add(tfp.layers.DistributionLambda(lf))
    return mdl

def dlambda_likelihood(y_true, dist):
    return -dist.log_prob(y_true)
