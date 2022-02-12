import logging
import sys
from matplotlib.ticker import Formatter
import numpy as np
from .utils import is_plot_visible, set_seed, timer
from .problem import BaseProblem
from .solvers import BaseMILP
from .surrogates import BaseSurrogate


class SearchLoop:

    def __init__(self, problem, cfg: dict):
        self.cfg = cfg
        self.problem: BaseProblem = problem
        self.verbosity = cfg['verbosity']
        self.logger = self.init_logging()
        self.iterations = cfg['iterations']
        self.starting_points = cfg['starting_points']
        surrogate_calss = cfg['surrogate_model']['type']
        milp_calss = cfg['milp_model']['type']
        self.surrogate_model: BaseSurrogate = surrogate_calss(problem, cfg['surrogate_model'], self.logger)
        self.milp_model: BaseMILP = milp_calss(problem, cfg['milp_model'], self.logger)
        self.init_dataset_callback = None
        self.iteration_callback = None

    @timer
    def init_dataset(self):
        set_seed()
        generated_x, generated_y = self.problem.get_dataset(self.starting_points)
        return generated_x, generated_y

    def init_logging(self):
        logger = logging.getLogger('emlopt')
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream)
        if self.verbosity == 0:
            logger.setLevel(logging.ERROR)
        elif self.verbosity == 1:
            logger.setLevel(logging.INFO)
        elif self.verbosity == 2:
            logger.setLevel(logging.DEBUG)
        else: 
            raise AttributeError(f"verbosity = {self.verbosity}")
        return logger

    @timer
    def step(self, iteration):
        learned_model, surrogate_runtime = self.surrogate_model.fit_surrogate(self.samples_x, self.samples_y, timer_logger=self.logger)

        if self.logger.level == logging.DEBUG and is_plot_visible():
            self.surrogate_model.plot_loss()
            self.surrogate_model.plot_predictions(learned_model, self.samples_x, self.samples_y)

        opt_x, milp_runtime = self.milp_model.optimize_acquisition_function(learned_model, self.samples_x, self.samples_y, timer_logger=self.logger)

        # check if repeated data point and query obj function
        opt_y: float 
        dists = np.sum(np.abs(self.samples_x - np.expand_dims(opt_x, 0)), axis=1)
        dmin, dargmin = np.min(dists), np.argmin(dists)
        if not self.problem.stocasthic and dmin <= 1e-6:
            self.logger.debug("Preventing query for an already known point !")
            opt_y = self.samples_y[dargmin]
        else:
            opt_y = self.problem.fun(opt_x)

        # not improve counter
        if opt_y < np.min(self.samples_y):
            self.milp_model.not_improve_iteration = 0
        else:
            self.milp_model.not_improve_iteration += 1
        self.milp_model.current_iteration = iteration

        # add new point to the dataset
        self.samples_x = np.concatenate((self.samples_x, np.expand_dims(opt_x, 0)))
        self.samples_y = np.append(self.samples_y, opt_y)

        self.logger.info(f"Iteration {iteration} objective value: {opt_y}")
        if self.iteration_callback is not None:
            self.iteration_callback({ 
                "x": opt_x,
                "y": opt_y,
                "surrogate_runtime": surrogate_runtime,
                "milp_runtime": milp_runtime
            })

    def run(self):
        try:
            (self.samples_x, self.samples_y), _ = self.init_dataset(timer_logger=self.logger)
            if self.init_dataset_callback is not None:
                self.init_dataset_callback()
        except Exception as e:
            self.logger.error("Error creating initial dataset", e)
            raise e

        try:
            for iteration in range(self.iterations):
                self.step(iteration, timer_logger=self.logger)

        except Exception as e:
            self.logger.error(f"Interrupted search due to exception\n{e}")

        ymin, yargmin = np.min(self.samples_y), np.argmin(self.samples_y)
        xmin = self.samples_x[yargmin]
        self.logger.info(f"Min found: {ymin} in {yargmin+1-self.starting_points} iterations")
        return xmin, ymin
