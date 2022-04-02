import tensorflow.keras
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
            print(klayer.__class__)
            raise ValueError('Unsupported layer type')
    # Return the network
    return net