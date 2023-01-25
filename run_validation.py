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


compute_target = ComputeTarget(ws, "yolo-cluster")

experiment = Experiment(workspace=ws, name="Validate-first-split")
script_args = [
    "--mount-point", mounted_dataset,
    "--data", "data-config/pano.yaml",
    "--weights", "weights/last-purple_boot_3l6p24vb.pt",
    "--batch-size", "1",
    "--max-det", "200",
    "--img", "8000",
    "--project", "outputs/runs/val",
    "--task", "val",
    "--save-txt"

]

script_config = ScriptRunConfig(
    source_directory=".",
    script="yolov5/val.py",
    arguments=script_args,
    environment=env,
    compute_target=compute_target,
)


run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
