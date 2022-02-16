import sys, os
sys.path.append('.')

from emlopt.config import EARLY_STOP
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

from problems.mccormick import objective

problem = build_problem("mccormick_2d", objective, ['real', 'real'], [[-1.5, 4], [-3, 4]])
search = SearchLoop(problem, EARLY_STOP)

wandb_cfg = WandbContext.get_defatult_cfg()
wandb_cfg['experiment_name'] = "plots"

with WandbContext(wandb_cfg, search):
    search.run()
