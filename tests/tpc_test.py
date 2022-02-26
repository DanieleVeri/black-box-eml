import sys, os
sys.path.append('.')

from emlopt.config import DEFAULT
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext

from problems.quantization.tpc import build_tpc, constraint_max_bits

tpc_obj = build_tpc()
problem = build_problem("tpc", tpc_obj, ["int"]*41, [[2, 8]]*41, constraint_max_bits)
search = SearchLoop(problem, DEFAULT)

wandb_cfg = WandbContext.get_defatult_cfg()
wandb_cfg['experiment_name'] = "TPC"

with WandbContext(wandb_cfg, search):
    search.run()
