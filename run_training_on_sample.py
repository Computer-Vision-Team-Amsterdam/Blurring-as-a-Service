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

dataset = Dataset.get_by_name(ws, "blur-sample")
mounted_dataset = dataset.as_mount(path_on_compute="data/sample")
compute_target = ComputeTarget(ws, "stronger-gpu")

experiment = Experiment(workspace=ws, name="Train-sample")
script_args = [
    "--data", "data-config/pano-sample.yaml",
    "--cfg", "weights/yolov5s.yaml",
    "--img", "2048",
    "--batch-size", "8",
    "--epochs", "2",
    "--project", "outputs/runs/train",
    "--cache"
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
