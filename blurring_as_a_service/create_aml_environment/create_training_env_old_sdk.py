from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings


def main():
    """
    Uncomment to create an environment using the old SDK.
    This code is commented because before running it, it's necessary to temporary install azureml-core package.
    """
    # settings = BlurringAsAServiceSettings.get_settings()
    # ws = Workspace.from_config()
    #
    # packages_and_versions_local_env = {
    #     ws.key: ws.version for ws in pkg_resources.working_set
    # }
    # packages_and_versions_local_env.pop("panorama")
    # packages_and_versions_local_env.pop("blurring-as-a-service")
    #
    # packages = [
    #     f"{package}=={version}"
    #     for package, version in packages_and_versions_local_env.items()
    # ]
    # env = Environment(settings["aml_experiment_details"]['env_name'])
    # env.docker.base_image = None
    # env.docker.base_dockerfile = "blur-environment.Dockerfile"
    # cd = CondaDependencies.create(
    #     python_version="3.9.16", pip_packages=packages
    # )
    # env.python.conda_dependencies = cd
    # env.register(workspace=ws)
    return


if __name__ == "__main__":
    settings = BlurringAsAServiceSettings.set_from_yaml("config.yml")
    main()
