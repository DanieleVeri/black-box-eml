from .convex_problem import ConvexRealProblem
from .integer_problem import IntegerProblem
from .base_problem import BaseProblem

from typing import Any, Callable, List
import numpy as np

def build_problem(
    name: str,
    fun: Callable[[np.ndarray],float],
    input_type: List[List],
    input_bounds: List[List],
    constraint_cb: Callable[[List],List] = None,
    stocasthic: bool = False):

    if 'int' in input_type:
        return IntegerProblem(name, fun, input_type, input_bounds, constraint_cb, stocasthic)
    else:
        return ConvexRealProblem(name, fun, input_type, input_bounds, constraint_cb, stocasthic)
