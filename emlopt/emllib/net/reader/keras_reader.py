import tensorflow as tf
import tensorflow.keras.layers as klayers

from .. import describe

def read_keras_sequential(kmodel):
    """ Import nueral network model from keras

    Casts neural network into custom representation, available at
    :obj:`eml.net.describe.DTNet`

    Parameters
    ----------
        kmodel : obj:`keras.models.Sequential`
            Trained keras neural network

    Returns
    -------
        Neural Network : :obj:`eml.net.describe.DNRNet`
            Neural Network with custom representation

    Raises
    ------
        ValueError
            If the layer type is not supported

    """
    # Build a DNR network model
    net = describe.DNRNet()
    # Add input layer
    kls = kmodel.layers
    layer = describe.DNRInput(input_shape=kls[0].input_shape[1:])
    net.add(layer)
    # Loop over the layers of the keras network
    for k, klayer in enumerate(kls):
        if klayer.__class__ == klayers.Dense:
            wgt, bias = klayer.get_weights()
            act = klayer.get_config()['activation']
            layer = describe.DNRDense(wgt, bias, act)
            net.add(layer)
        elif klayer.__class__ != tensorflow.keras.layers.InputLayer:
            raise ValueError(f'Unsupported layer type: {klayer.__class__}')
    # Return the network
    return net

def read_keras_probabilistic_sequential(model):
    in_shape = model.input_shape[1]
    mdl_no_dist = tf.keras.Sequential()
    mdl_no_dist.add(tf.keras.layers.Input(shape=(in_shape,), dtype='float32'))
    for i in range(len(model.layers)-2):
        w = model.layers[i].weights[1].shape[0]
        mdl_no_dist.add(tf.keras.layers.Dense(w, activation='relu'))
    mdl_no_dist.add(tf.keras.layers.Dense(2, activation='linear'))

    mdl_no_dist.set_weights(model.get_weights())

    nn = read_keras_sequential(mdl_no_dist)
    return nn
