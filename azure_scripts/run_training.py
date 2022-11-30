import os
import json
from azureml.core import (
    ComputeTarget,
    Dataset,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)

ws = Workspace.from_config()
env = Environment.from_dockerfile("blurring-env", "data-prep/laurens/blur-environment")
dataset = Dataset.get_by_name(ws, "blur_v1")

mounted_dataset = dataset.as_mount(path_on_compute="blurring-project/")
compute_target = ComputeTarget(ws, "stronger-gpu")
experiment = Experiment(workspace=ws, name="Create-dataset-from-locations")
