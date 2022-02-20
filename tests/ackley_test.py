import sys, os
sys.path.append('.')

from emlopt.config import DEFAULT, INCREMENTAL_TEST
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

from problems.ackley import build_ackley, constraint_scbo

problem = build_problem("ackley_10d_cst", *build_ackley(10), constraint_scbo)
search = SearchLoop(problem, INCREMENTAL_TEST)

wandb_cfg = WandbContext.get_defatult_cfg()
wandb_cfg['experiment_name'] = "INCREMENTAL"

with WandbContext(wandb_cfg, search):
    search.run()
