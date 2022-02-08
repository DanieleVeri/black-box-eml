import numpy as np
import tensorflow as tf

from .base_surrogate import BaseSurrogate
from ..utils import min_max_scale_in, min_max_scale_out


class StopCICB(tf.keras.callbacks.Callback):

    def __init__(self, threshold, logger, *args, **kwargs):
        super(StopCICB, self).__init__(*args, **kwargs)
        self.threshold = threshold
        self.logger = logger
        self.x = None

    def on_epoch_end(self, epoch, logs={}):
        prob_pred = self.model(self.x)
        std_pred = prob_pred.stddev().numpy().ravel()
        if epoch % 100 == 0:
            self.logger.debug(f"epoch {epoch} - mean stddev {np.mean(std_pred)}")
        if np.mean(std_pred) < self.threshold:
            self.logger.debug(f"break condition on epoch {epoch} with mean stddev {np.mean(std_pred)}")
            self.model.stop_training = True


class StopCI(BaseSurrogate):

    def __init__(self, *args, **kwargs):
        super(StopCI, self).__init__(*args, **kwargs)
        self.ci_threshold = self.cfg['ci_threshold']
        stop_ci = StopCICB(self.ci_threshold, self.logger)
        self.cb = [stop_ci]

    def train(self, keras_mdl, x, y):
        bs = self.batch_size if self.batch_size else x.shape[0]

        norm_x = min_max_scale_in(x, np.array(self.problem.input_bounds))
        norm_y = min_max_scale_out(y, y)

        self.cb[0].x = norm_x

        hstory = keras_mdl.fit(
            norm_x, norm_y,
            batch_size=bs, epochs=self.epochs, verbose=0, callbacks=self.cb)

        return hstory
