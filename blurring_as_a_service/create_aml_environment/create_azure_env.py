from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


def main():
    """
    This file creates an AML environment.
    This code is commented because before running it, it's necessary to temporary install azureml-core package.
    """
    settings = BlurringAsAServiceSettings.get_settings()

    aml_interface = AMLInterface()

    custom_packages = {
        "panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2"
    }
    aml_interface.create_aml_environment(
        env_name=settings["aml_experiment_details"]["env_name"],
        project_name="blurring-as-a-service",
        submodules=["yolov5"],
        custom_packages=custom_packages,
    )


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
