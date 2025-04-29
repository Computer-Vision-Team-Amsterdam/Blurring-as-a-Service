import logging
import os

from aml_interface.aml_interface import AMLInterface

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
    aml_interface.create_aml_environment(
        env_name=settings["aml_experiment_details"]["env_name"],
        build_context_path="blurring_as_a_service/create_aml_environment",
        dockerfile_path="blur-environment.Dockerfile",
        build_context_files=["pyproject.toml"],
    )


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
