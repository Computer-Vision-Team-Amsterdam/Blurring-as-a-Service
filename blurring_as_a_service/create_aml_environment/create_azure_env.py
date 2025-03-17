import logging
import os
import shutil

from aml_interface.aml_interface import AMLInterface
from azure.ai.ml.entities import BuildContext, Environment

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

logger = logging.getLogger(__name__)


def delete_file(file_path):
    try:
        os.remove(file_path)
        logger.info(f"{file_path} has been deleted.")
    except FileNotFoundError:
        logger.error(f"{file_path} does not exist.")
    except Exception as e:
        logger.error(f"Failed to remove file '{file_path}': {str(e)}")
        raise Exception(f"Failed to remove file '{file_path}': {e}")


def main():
    """
    This file creates an AML environment.
    """
    settings = BlurringAsAServiceSettings.get_settings()
    aml_interface = AMLInterface()
    # aml_interface.create_aml_environment(
    #     env_name=settings["aml_experiment_details"]["env_name"],
    #     build_context_path="blurring_as_a_service/create_aml_environment",
    #     dockerfile_path="blur-environment.Dockerfile",
    #     build_cluster="cpu-cluster",
    # )

    ws = aml_interface.ml_client.workspaces.get(
        name=aml_interface.ml_client.workspace_name
    )
    ws.image_build_compute = "cpu-cluster"

    shutil.copyfile(
        "poetry.lock", "blurring_as_a_service/create_aml_environment/poetry.lock"
    )
    shutil.copyfile(
        "pyproject.toml", "blurring_as_a_service/create_aml_environment/pyproject.toml"
    )
    env = Environment(
        name=settings["aml_experiment_details"]["env_name"],
        build=BuildContext(
            path="blurring_as_a_service/create_aml_environment",
            dockerfile_path="blur-environment.Dockerfile",
        ),
    )
    aml_interface.ml_client.environments.create_or_update(env)
    delete_file("blurring_as_a_service/create_aml_environment/poetry.lock")
    delete_file("blurring_as_a_service/create_aml_environment/pyproject.toml")


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
