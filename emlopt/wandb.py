import pickle
import numpy as np
import wandb
import tempfile
from .search_loop import SearchLoop


class WandbContext:

    @staticmethod
    def get_defatult_cfg():
        return {
            "project": "eml",
            "experiment_name": "new_experiment",
            "run_id": None
        }

    def __init__(self, wandb_config: dict, search: SearchLoop):
        self.wandb_config = wandb_config
        self.search = search

    def __enter__(self):
        self.init_wandb(self.wandb_config, self.search.cfg)
        self.search.init_dataset_callback = self.on_init
        self.search.iteration_callback = self.on_end_iteration
        self.search.milp_model.solution_callback = self.on_solution

    def __exit__(self, exc_type, exc_value, tb):
        try:
            self.save_points()
        finally:
            wandb.finish()

    def init_wandb(self, wandb_config: dict, search_config: dict):
        if wandb_config['run_id'] is not None:
            wandb.init(project=wandb_config['project'], id=wandb_config['run_id'], resume='allow')
        else:
            wandb.init(project=wandb_config['project'], name=wandb_config['experiment_name'])
        wandb.config.update(search_config)

    def on_init(self, obj={}):
        for j in range(self.search.starting_points):
            wandb.log({
                "x": self.search.samples_x[j],
                "y": self.search.samples_y[j]
            })

    def on_solution(self, main_vars, all_vars):
        wandb.log(main_vars, commit=False)

    def on_end_iteration(self, obj={}):
        if self.search.verbosity == 2:
            wandb.log({"train_loss": wandb.Image(f'{tempfile.gettempdir()}/train_loss.png')}, commit=False)
            if self.search.problem.input_shape <= 2:
                wandb.log({"train_predictions": wandb.Image(f'{tempfile.gettempdir()}/chart.png')}, commit=False)
        wandb.log({"y_min": np.min(self.search.samples_y)}, commit=False)
        wandb.log(obj)

    def save_points(self):
        wandb.log({
            "min_found": np.min(self.search.samples_y),
            "n_iterations": np.argmin(self.search.samples_y)+1-self.search.starting_points
        })
        dump_points = list(zip(self.search.samples_x, self.search.samples_y))
        with open('points.pkl', 'wb') as f:
            pickle.dump(dump_points, f)
        artifact = wandb.Artifact('datapoints', type='points')
        artifact.add_file("points.pkl")
        wandb.log_artifact(artifact)
