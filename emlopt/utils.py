import os
import time
import logging
import functools
import random
import numpy as np
import tensorflow as tf


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger = kwargs.pop('timer_logger')
        logger.debug(f"Started {func.__name__!r}")
        start_time = time.perf_counter()   
        value = func(*args, **kwargs)
        end_time = time.perf_counter()     
        run_time = end_time - start_time   
        logger.debug(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time
    return wrapper_timer

def set_seed(seed: int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def is_plot_visible():
    is_x_server = 'DISPLAY' in os.environ
    try:
        is_jupyter = 'ipykernel' in str(get_ipython())
        is_colab = 'google.colab' in str(get_ipython())
        is_notebook = is_jupyter or is_colab
    except:
        is_notebook = False
    return is_x_server or is_notebook

def min_max_scale_in(values: np.array, bounds: np.array) -> np.array:
    return (values-bounds[:, 0]) / (bounds[:, 1]-bounds[:, 0])

def min_max_scale_out(values: np.array, samples: np.array) -> np.array:
    min_bound, max_bound = np.min(samples), np.max(samples)
    return (values-min_bound) / (max_bound-min_bound)

def min_max_restore_out(values: np.array, samples: np.array, stddev: bool = False) -> np.array:
    min_bound, max_bound = np.min(samples), np.max(samples)
    if stddev:
        return values*(max_bound-min_bound)
    return (max_bound-min_bound)*values + min_bound
