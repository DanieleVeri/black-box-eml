# EMLOpt
## Emlpirical Model Learning for Contrained Black Box optimization
Daniele Verì

[Development status](https://github.com/LIA-UniBo/eml-opt/projects/1)

## Usage
```python
from emlopt.search_loop import SearchLoop
from emlopt.problem import build_problem
from emlopt.wandb import WandbContext
from emlopt.config import DEFAULT

# Define the objecive function
def objective_function(x):
    value = f(x[0], x[1])
    return value

# Define the domain constraints
def constraint(backend, milp_model, x):
    return [ [x[0]**2 + x[1]**2 <= 1 , "constraint name"] ]

# Define the decision variable bounds and types
bounds = [[-5,5], [3, 10]]
types = ['int', 'real']

# Create problem object
problem = build_problem(
    "test function",
    objective_function,
    bounds,
    types,
    constraint)

# Create search object
search = SearchLoop(problem, DEFAULT)

# Start the search
search.run()

# OPTIONAL: Use the Weight and Biases context in order to log metrics and results
with WandbContext(WandbContext.get_defatult_cfg(), search):
    search.run()
```

## Configuration
Default configuraition object definition:
```python
{
    "verbosity": 2,

    "iterations": 100,
    "starting_points": 100,

    "surrogate_model":
    {
        "type": "stop_ci",
        "epochs": 999,
        "learning_rate": 5e-3,
        "weight_decay": 1e-4,
        "batch_size": None,
        "depth": 1,
        "width": 200,
        "ci_threshold": 5e-2,
    },

    "milp_model":
    {
        "type": "simple_dist",
        "backend": "cplex",
        "lambda_ucb": 1,
        "solver_timeout": 120,
    }
}
```
| Field | Domain | Description |
|-|-|-|
| verbosity | 0, 1, 2 | set the amount of log to be produced
| iterations | positive integer | the number of search steps
| starting_points | positive integer | the number of initial samples
| surrogate_model.type | stop_ci, early_stop, uniform_noise  | the surrogate model training procedure
| surrogate_model.epochs | positive integer  | the max number of epochs
| surrogate_model.learning_rate | positive real | the Adam learning rate
| surrogate_model.weight_decay | positive real | the Adam weight decay
| surrogate_model.batch_size | positive int or None | the batch size, None means single batch
| surrogate_model.depth | positive int | the NN depth
| surrogate_model.width | positive int | the NN width
| surrogate_model.ci_threshold | positive real | the confidence interval threshold for the stop_ci surrogate model
| milp_model.type | ucb, simple_dist, incremental_dist, speedup_dist, lns_dist | the milp solver model
| milp_model.backend | cplex, ortools | the milp solver backend
| milp_model.bound_propagation | ibr, milp, both | the bound propagation algorithm; both performs ibr first and then milp
| milp_model.lambds_ucb | positive real | the coefficient that balances UCB exploration/exploitation
| milp_model.solver_timeout | positive integer | the milp solver timeout in seconds


## Backends
- **cplex**: The IBM CPLEX MIL(Q)P solver. Can be used only for personal or accademic purposes.
- **ortools**: The Google OrTools solver. Can be used without limits but cannot handle quadratic contraints, also is much slower than CPLEX.

## Folder
- **tests**: Contains the unit tests that validate the proper functionality of the EML library and the optimizaiton loop with both the backends.
- **experiments**: Contains the script that launch the search on hard optimization problems while logging the results on Weights and Biases.
- **notebooks**: Contains the notebooks for display easily the results.

## Debug
The Dockerfile is configured to allow the remote debugging of the python code with VS Code.
Just run `./launch_debug <path_to_file>` and then click on 'Remote debug attach' on the IDE.
