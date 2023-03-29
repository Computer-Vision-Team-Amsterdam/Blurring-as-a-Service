import pkg_resources
from azureml.core import Environment, Workspace
from azureml.core.conda_dependencies import CondaDependencies

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings


def main():
    """
    This file creates an AML environment using the old SDKv1.
    This code is commented because before running it, it's necessary to temporary install azureml-core package.
    """
    settings = BlurringAsAServiceSettings.get_settings()
    ws = Workspace.from_config()

    packages_and_versions_local_env = {
        ws.key: ws.version for ws in pkg_resources.working_set
    }
    packages_and_versions_local_env.pop("panorama")
    # packages_and_versions_local_env.pop("blurring-as-a-service")

    packages = [
        f"{package}=={version}"
        for package, version in packages_and_versions_local_env.items()
    ]
    env = Environment(settings["aml_experiment_details"]["env_name"])
    env.docker.base_image = None
    env.docker.base_dockerfile = "blur-env-cuda11.8.Dockerfile"
    cd = CondaDependencies.create(python_version="3.9.16", pip_packages=packages)
    env.python.conda_dependencies = cd
    env.register(workspace=ws)
    return


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
