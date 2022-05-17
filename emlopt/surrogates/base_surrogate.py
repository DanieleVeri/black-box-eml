import logging
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from ..tfp import build_probabilistic_regressor, dlambda_likelihood
from ..utils import is_plot_visible, min_max_scale_in, min_max_restore_out, timer
from ..problem import BaseProblem


class BaseSurrogate:

    def __init__(self, problem, surrogate_cfg, logger):
        self.problem: BaseProblem = problem
        self.cfg = surrogate_cfg
        self.logger = logger
        self.epochs = surrogate_cfg['epochs']
        self.lr = surrogate_cfg['learning_rate']
        self.weight_decay = surrogate_cfg['weight_decay']
        self.depth = surrogate_cfg['depth']
        self.width = surrogate_cfg['width']
        self.batch_size = surrogate_cfg['batch_size']

        self.loss_history = None

    def train(self, keras_mdl, x, y):
        raise NotImplementedError

    @timer
    def fit_surrogate(self, x, y):
        self.logger.info(f"{self.__class__.__name__} surrogate:")
        optimizer = tfa.optimizers.AdamW(
            weight_decay=self.weight_decay, learning_rate=self.lr)

        keras_mdl = build_probabilistic_regressor(
            self.problem.input_shape, self.depth, self.width)

        keras_mdl.compile(optimizer=optimizer, loss=dlambda_likelihood)
        if self.logger.level == logging.DEBUG:
            keras_mdl.summary()

        self.loss_history = self.train(keras_mdl, x ,y)

        return keras_mdl

    def plot_loss(self):
        plt.plot(self.loss_history.history["loss"])
        plt.savefig('train_loss.png')
        if is_plot_visible(): plt.show()
        else: plt.close()

    def plot_predictions(self, keras_mdl, samples_x, samples_y):
        if self.problem.input_shape <= 2:
            x, y = self.problem.get_grid(100)
            scaled_x = min_max_scale_in(x, np.array(self.problem.input_bounds))

            prob_pred = keras_mdl(scaled_x)

            pred = prob_pred.mean().numpy().ravel()
            pred = min_max_restore_out(pred, samples_y)
            std_pred = prob_pred.stddev().numpy().ravel()
            std_pred = min_max_restore_out(std_pred, samples_y, stddev=True)

        # 1D domain
        if self.problem.input_shape == 1:
            fig = plt.figure(figsize=(15, 10))
            plt.xlim(self.problem.input_bounds[0])
            x = np.squeeze(x)
            plt.plot(x, y, c="grey")
            plt.plot(x, pred)
            plt.fill_between(x, pred-std_pred, pred+std_pred, alpha=0.3, color='tab:blue', label='+/- std')
            plt.scatter(samples_x, samples_y, c="orange")
            plt.legend(["GT", "predicted mean", "predicted CI", "samples"], prop={'size': 14})
            plt.savefig('chart.png')
            if is_plot_visible(): plt.show()
            else: plt.close()

        # 2D domain
        elif self.problem.input_shape == 2:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(
                samples_x[:, 0], samples_x[:, 1], samples_y, color="orange")
            ax.scatter(x[:, 0], x[:, 1], y, alpha=0.15, color="lightgrey")
            ax.scatter(x[:, 0], x[:, 1], pred, alpha=0.3)
            ax.scatter(x[:, 0], x[:, 1], pred-std_pred, alpha=0.3, color="lightblue")
            ax.scatter(x[:, 0], x[:, 1], pred+std_pred, alpha=0.3, color="lightblue")
            ax.view_init(elev=15, azim=60)
            plt.legend(["samples", "GT", "predicted mean", "predicted CI"], prop={'size': 14})
            plt.savefig('chart.png')
            if is_plot_visible(): plt.show()
            else: plt.close()

        else:
            self.logger.debug("Plot not available for high dimensional domains.")
            self.logger.debug(f"X:\n{samples_x}\nY:\n{samples_y}")

            scaled_samples_x = min_max_scale_in(samples_x,np.array(self.problem.input_bounds))
            prob_pred = keras_mdl(scaled_samples_x)
            mu = min_max_restore_out(prob_pred.mean().numpy().ravel(), samples_y)

            self.logger.debug(f"mu pred\n{mu}")
            self.logger.debug(f"diff\n{samples_y-mu}")
