import pkg_resources
from azureml.core import (
    ComputeTarget,
    Dataset,
    Environment,
    Experiment,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()
packages_and_versions_local_env = dict(
    tuple(str(ws).split()) for ws in pkg_resources.working_set
)
packages = [
    f"{package}=={version}"
    for package, version in packages_and_versions_local_env.items()
]

env = Environment("blurring-env")
env.docker.base_image = None
env.docker.base_dockerfile = "blur-environment.Dockerfile"
cd = CondaDependencies.create(
    python_version="3.8.14", pip_packages=packages
)

env.python.conda_dependencies = cd
env.register(workspace=ws)
dataset = Dataset.get_by_name(ws, "ORBS-base-first-split")

mounted_dataset = dataset.as_mount(path_on_compute="first-split")
compute_target = ComputeTarget(ws, "stronger-gpu")
experiment = Experiment(workspace=ws, name="Train-with-first-split")


script_args = [
    "--data", "data/pano.yaml",
    "--img", "4000",
]
script_config = ScriptRunConfig(
    source_directory="./yolov5",
    script="train.py",
    arguments=script_args,
    environment=env,
    compute_target=compute_target,
)
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
