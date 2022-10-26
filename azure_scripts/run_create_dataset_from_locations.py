"""
This module contains functionality to run a training script on Azure.
"""
from azureml.core import (
    ComputeTarget,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)

ws = Workspace.from_config()
env = Environment.from_dockerfile("blurring-env", "blur-environment")

compute_target = ComputeTarget(ws, "laurens-cpu")
experiment = Experiment(workspace=ws, name="Create-dataset-from-locations")

script_config = ScriptRunConfig(
    source_directory=".",
    script="create_dataset_from_locations.py",
    environment=env,
    compute_target=compute_target,
)
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
