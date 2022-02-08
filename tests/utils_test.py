import sys, os
sys.path.append('.')

from emlopt.config import DEFAULT
from emlopt.search_loop import SearchLoop
from emlopt.problem import ConvexRealProblem
from emlopt.wandb import WandbExperiment

from problems.ackley import build_ackley, constraint_scbo


problem = ConvexRealProblem("test", *build_ackley(10), constraint_scbo)
search = SearchLoop(problem, DEFAULT)

wandb_cfg = WandbExperiment.get_defatult_cfg()
wandb_cfg['experiment_name'] = "navajo joe"

with WandbExperiment(wandb_cfg, search):
    search.run()
