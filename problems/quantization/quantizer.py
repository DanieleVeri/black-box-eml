import os
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import default_8bit_quantize_scheme, default_8bit_quantize_registry, default_8bit_quantizers
from tensorflow_model_optimization.quantization.keras.quantizers import LastValueQuantizer, MovingAverageQuantizer
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer as ql
from tensorflow_model_optimization.quantization.keras import quantize_annotate_model, QuantizeConfig


class ConvWeightsQuantizer(LastValueQuantizer):
  """Quantizer for handling weights in Conv2D/DepthwiseConv2D layers."""
  def __init__(self, bits):
        super(ConvWeightsQuantizer, self).__init__(
            num_bits=bits, per_axis=True, symmetric=True, narrow_range=True)
  def build(self, tensor_shape, name, layer):
    min_weight = layer.add_weight(
        name + '_min',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(-6.0),
        trainable=False)
    max_weight = layer.add_weight(
        name + '_max',
        shape=(tensor_shape[-1],),
        initializer=tf.keras.initializers.Constant(6.0),
        trainable=False)
    return {'min_var': min_weight, 'max_var': max_weight}

class QConf(default_8bit_quantize_registry.Default8BitQuantizeConfig):
    def __init__(self, bits, conv, *args, **kwargs):
        super(QConf, self).__init__(*args, **kwargs)
        self.bits = bits
        if conv:
            self.weight_quantizer = ConvWeightsQuantizer(bits)
        else:
            self.weight_quantizer = LastValueQuantizer(
                num_bits=self.bits, per_axis=False, symmetric=True, narrow_range=True)
        self.activation_quantizer = MovingAverageQuantizer(
            num_bits=self.bits, per_axis=False, symmetric=False, narrow_range=False)

class QAct(default_8bit_quantize_registry.Default8BitActivationQuantizeConfig):
    def __init__(self, bits, *args, **kwargs):
        super(QAct, self).__init__(*args, **kwargs)
        self.bits = bits

    def get_output_quantizers(self, layer):
        self._assert_activation_layer(layer)
        if not hasattr(layer.activation, '__name__'):
            raise ValueError('Activation {} not supported by '
                            'Default8BitActivationQuantizeConfig.'.format(
                                layer.activation))
        if layer.activation.__name__ in ['relu', 'swish']:
            return [MovingAverageQuantizer(
            num_bits=self.bits, per_axis=False, symmetric=False, narrow_range=False)]
        elif layer.activation.__name__ in ['linear', 'softmax', 'sigmoid', 'tanh']:
            return []
        raise ValueError('Activation {} not supported by '
                        'Default8BitActivationQuantizeConfig.'.format(
                            layer.activation))

class QReg(default_8bit_quantize_registry.Default8BitQuantizeRegistry):
    def __init__(self, bitlist, *args, **kwargs):
        super(QReg, self).__init__(*args, **kwargs)
        self.bitlist = bitlist
        self.counter = -1

    def get_quantize_config(self, layer):
        self.counter += 1
        quantize_info = self._get_quantize_info(layer.__class__)
        if layer.name.startswith('activation'):
            return QAct(self.bitlist[self.counter])
        return QConf(self.bitlist[self.counter], 
                     layer.name.startswith('conv'),  # enable ConvWeightsQuantizer
                     quantize_info.weight_attrs,
                     quantize_info.activation_attrs,
                     quantize_info.quantize_output)

class QScheme(default_8bit_quantize_scheme.Default8BitQuantizeScheme):
    def __init__(self, bitlist, *args, **kwargs):
        super(QScheme, self).__init__(*args, **kwargs)
        self.bitlist = bitlist

    def get_quantize_registry(self):
        return QReg(self.bitlist)

def download_pretrained_model():
    os.system("mkdir pretrained_resnet18")
    os.system("mkdir pretrained_resnet18/variables")
    os.system("wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6NTU2NTg0NjE=/d9d4d8f866df84014e528bb3c5617816 -O  pretrained_resnet18/variables/variables.data-00000-of-00001")
    os.system("wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6NTU2NTg0NjE=/4901af0e55327757ca7d7380b353279f -O  pretrained_resnet18/variables/variables.index")
    os.system("wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6NTU2NTg0NjE=/0a1a30ebb8498c7adaab17365283b563 -O  pretrained_resnet18/keras_metadata.pb")
    os.system("wget https://api.wandb.ai/artifactsV2/gcp-us/veri/QXJ0aWZhY3Q6NTU2NTg0NjE=/7c8a4682f521bac78f8a89b70342675b -O  pretrained_resnet18/saved_model.pb")
    model = tf.keras.models.load_model("pretrained_resnet18")
    
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    dataset_mean = train_images.mean(axis=(0,1,2))
    dataset_std = train_images.std(axis=(0,1,2))
    test_images = (test_images - dataset_mean) / dataset_std
    _, baseline_model_accuracy = model.evaluate(
        test_images, test_labels, verbose=0)
    print('Baseline test accuracy:', baseline_model_accuracy)
    return model

