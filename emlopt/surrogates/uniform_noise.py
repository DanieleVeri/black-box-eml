import numpy as np
import tensorflow as tf

from .stop_ci import StopCI
from ..utils import min_max_scale_in, min_max_scale_out


class UniformNoise(StopCI):

    def __init__(self, *args, **kwargs):
        super(UniformNoise, self).__init__(*args, **kwargs)
        self.noise_ratio = self.cfg['noise_ratio']
        self.sample_weight = self.cfg['sample_weights']

    def train(self, keras_mdl, x, y):
        noise_points = self.noise_ratio * x.shape[0]
        y_range = np.max(np.abs(y))*2                              
        noise_x = self.problem.get_dataset(noise_points, False)
        noise_y = (np.random.rand(noise_points)-0.5)*y_range
        x_aug_samples = np.concatenate((x, noise_x))
        y_aug_samples = np.concatenate((y, noise_y))  

        sw = np.ones_like(x_aug_samples)  
        sw[:x.shape[0]] = self.sample_weight * self.noise_ratio

        bs = self.batch_size if self.batch_size else x_aug_samples.shape[0]

        norm_x = min_max_scale_in(x_aug_samples, np.array(self.problem.input_bounds))
        norm_y = min_max_scale_out(y_aug_samples, y_aug_samples)

        self.cb[0].x = norm_x

        hstory = keras_mdl.fit(
            norm_x, norm_y,
            batch_size=bs, epochs=self.epochs, 
            verbose=0, callbacks=self.cb,  sample_weight=sw)

        return hstory