from .base_milp import BaseMILP
from .speedup_dist import SpeedupDist
from .simple_dist import SimpleDist
from .incremental_dist import IncrementalDist
from .lns_dist import LNSDist
from .ucb import UCB

def get_solver_class(solver):
    if solver == "ucb":
        return UCB
    if solver == "simple_dist":
        return SimpleDist
    if solver == "incremental_dist":
        return IncrementalDist
    if solver == "speedup_dist":
        return SpeedupDist
    if solver == "lns_dist":
        return LNSDist
