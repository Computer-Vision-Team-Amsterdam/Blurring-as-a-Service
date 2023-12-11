from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings
from blurring_as_a_service.utils.aml_interface import AMLInterface


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
    )


if __name__ == "__main__":
    BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
