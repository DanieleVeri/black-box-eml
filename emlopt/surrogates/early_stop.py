import numpy as np
import tensorflow as tf

from .base_surrogate import BaseSurrogate
from ..utils import min_max_scale_in, min_max_scale_out, timer


class EarlyStop(BaseSurrogate):

    def __init__(self, *args, **kwargs):
        super(EarlyStop, self).__init__(*args, **kwargs)
        self.num_val_points = self.cfg['num_val_points']
        self.patience = self.cfg['patience']

        (self.x_val, self.y_val), _ = self.init_validation_set(timer_logger=self.logger)

        self.cb = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)]

    @timer
    def init_validation_set(self):
        return self.problem.get_dataset(self.num_val_points)

    def train(self, keras_mdl, x, y):
        bs = self.batch_size if self.batch_size else x.shape[0]

        norm_x = min_max_scale_in(x, np.array(self.problem.input_bounds))
        norm_y = min_max_scale_out(y, y)

        val_x = min_max_scale_in(self.x_val, np.array(self.problem.input_bounds))
        val_y = min_max_scale_out(self.y_val, y)

        hstory = keras_mdl.fit(norm_x, norm_y, 
            validation_data=(val_x, val_y),
            batch_size=bs, epochs=self.epochs, 
            verbose=0, callbacks=self.cb)

        return hstory
