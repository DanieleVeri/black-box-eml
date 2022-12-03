# EMLOpt
## Emlpirical Model Learning for Contrained Black Box optimization

Master's thesis in AI under the supervision of prof. Michele Lombardi and Andrea Borghesi.

Based on work: Michele Lombardi, Michela Milano, and Andrea Bartolini. Empirical decision model learning. Artificial intelligence, 244, 2017-03

This work employs a NN as surrogate model that is embedded into a MILP prescriptive model.
The prescriptive model is then enriched with the domain constraints and optimize the acquisition function.

## Commands
- Start jupyter server: `docker-compose up`
- Run all tests: `docker-compose run development tests`
- Launch debug server on file: `docker-compose run --service-ports debug <path_to_file>`

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
        "bound_propagation": "both",
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
| milp_model.bound_propagation | ibr, milp, both, domain | the bound propagation algorithm
| milp_model.lambds_ucb | positive real | the coefficient that balances UCB exploration/exploitation
| milp_model.solver_timeout | positive integer | the milp solver timeout in seconds

## Bound propagation
- Interval Bound Reasoning: fast and coarse method to obtain preliminary bounds
- MILP: compute per neuron bounds by maximizing/minimizing the pre activation value
- Both: Performs IBR and then MILP
- Domain: Like the 'both' method but integrates also domain specific contraints, resulting in a slower bound propagation that computes tighter bounds.

## Backends
- **cplex**: The IBM CPLEX MIL(Q)P solver. Can be used only for personal or accademic purposes.
- **ortools**: The Google OrTools solver. Can be used without limits but **cannot handle quadratic contraints**, also is much slower than CPLEX.

## Folder
- **tests**: Contains the unit tests that validate the proper functionality of the EML library and the optimizaiton loop with both the backends.
- **experiments**: Contains the script that launch the search on hard optimization problems while logging the results on Weights and Biases.
- **notebooks**: Contains the notebooks for display easily the results.

## Debug
The Dockerfile is configured to allow the remote debugging of the python code with VS Code.
In order to run the debug server, open a shell inside the docker container, then `cd tests` and run `./launch_debug <fine_name>.`
Finally click on 'Remote debug attach' in the IDE.
