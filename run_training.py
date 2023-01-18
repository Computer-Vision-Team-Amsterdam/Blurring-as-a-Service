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


# Get or create environment
try:
    env = Environment.get(ws, 'blurring-env')
except Exception:
    print("Environment not found. Building environment from blur-environment.Dockerfile")
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
mounted_dataset = dataset.as_mount(path_on_compute="data/first-split")
compute_target = ComputeTarget(ws, "stronger-gpu")

experiment = Experiment(workspace=ws, name="Train-with-first-split")
script_args = [
    "--mount-point", mounted_dataset,  # this is needed otherwise the mounted folder cannot be found
    "--data", "yolov5/data/pano.yaml",
    "--cfg", "yolov5/models/yolov5s.yaml",
    "--img", "2048",
    "--batch-size", "8",
    "--epochs", "100",
    "--save-period", "40",
    "--project", "outputs/runs/train",
    "--cache",
    "--resume",
    "--weights", "yolov5/models/last-willing_plum_1t32npvv.pt"
]

script_config = ScriptRunConfig(
    source_directory=".",
    script="yolov5/train.py",
    arguments=script_args,
    environment=env,
    compute_target=compute_target,
)


run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
