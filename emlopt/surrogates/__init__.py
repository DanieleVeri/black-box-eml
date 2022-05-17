from .base_surrogate import BaseSurrogate
from .stop_ci import StopCI
from .early_stop import EarlyStop
from .uniform_noise import UniformNoise

def get_surrogate_class(surrogate):
    if surrogate == "stop_ci":
        return StopCI
    if surrogate == "early_stop":
        return EarlyStop
    if surrogate == "uniform_noise":
        return UniformNoise
