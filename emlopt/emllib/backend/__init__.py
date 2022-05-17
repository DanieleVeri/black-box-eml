from .base import Backend
from .cplex_backend import CplexBackend
from .ortool_backend import OrtoolsBackend

def get_backend(backend):
    if backend == "ortools":
        return OrtoolsBackend()
    if backend == "cplex":
        return CplexBackend()
