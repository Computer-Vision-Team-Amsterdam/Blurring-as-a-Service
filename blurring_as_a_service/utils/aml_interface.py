import logging
import os
from typing import Dict, List

import pkg_resources
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

logger = logging.getLogger(__name__)


class AMLInterface:
    """This class provides an interface to interact with Azure ML.

    Attributes
    ----------
    workspace :
        Instance of :class:`azureml.core.Workspace`
    """

    def __init__(self):
        """Initiate AMLInterface based on the Azure config.json file."""
        self.workspace = MLClient.from_config(self._connect())
        logger.info(
            f"Retrieved the following workspace: {self.workspace.workspace_name}"
        )

    @staticmethod
    def _connect():
        """
        Connects to the ML workspace and other components using the Managed Identity of the workspace
        """
        try:
            credential = DefaultAzureCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            # This will open a browser page for
            logger.info("Using InteractiveBrowserCredential login...")
            credential = InteractiveBrowserCredential()
        return credential

    def create_aml_environment(
        self,
        env_name: str,
        project_name: str,
        submodules: List[str] = [],
        custom_packages: Dict[str, str] = {},
    ) -> Environment:
        """Creates an AML environment based on the mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04 image.
        Installs the pip packages present in the env where the code is run.

        Parameters
        ----------
        env_name : str
            Name to give to the new environment.
        project_name: str
            Name of the project to be removed from the dependencies in case locally you are using Poetry.
        submodules : List[str]
            Packages that are actually submodules and not pip installed.
        custom_packages: Dict[str, str]
            Custom packages to remove from the local dependencies list and install on the AzureML environment.
            Example: {"panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2"}

        Returns
        -------
        : Environment
            Created environment.
        """
        self._create_environment_yml(project_name, submodules, custom_packages)
        env = Environment(
            name=env_name,
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="environment.yml",
        )
        self.workspace.environments.create_or_update(env)
        self._delete_environment_yml()
        return env

    @staticmethod
    def _create_environment_yml(
        project_name: str,
        submodules: List[str] = [],
        custom_packages: Dict[str, str] = {},
    ):
        """
        Retrieves all packages currently installed in the local venv used to execute the code,
        and creates a conda environment yaml file to be used to install the packages on AzureML env.

        Parameters
        ----------
        project_name: str
            Name of the project to be removed from the dependencies in case locally you are using Poetry.
        submodules : List[str]
            Packages that are actually submodules and not pip installed.
        custom_packages: Dict[str, str]
            Custom packages to remove from the local dependencies list and install on the AzureML environment.
            Example: {"panorama": "git+https://github.com/Computer-Vision-Team-Amsterdam/panorama.git@v0.2.2"}
        """
        packages_and_versions_local_env = {
            ws.key: ws.version for ws in pkg_resources.working_set
        }
        packages_and_versions_local_env.pop(project_name)
        for custom_package in custom_packages.keys():
            packages_and_versions_local_env.pop(custom_package)
        packages = [
            f"    - {key}=={value}" if key not in submodules else f"    - {key}"
            for key, value in packages_and_versions_local_env.items()
        ]

        for custom_package in custom_packages.values():
            packages.append(f"    - {custom_package}")
        with open("environment.yml", "w") as env_file:
            env_file.write("dependencies:\n")
            env_file.write("  - python=3.9.*\n")
            env_file.write("  - pip:\n")
            env_file.write("\n".join(packages))

    @staticmethod
    def _delete_environment_yml():
        os.remove("environment.yml")

    def submit_command_job(self, job):
        """
        Examples
        ________
        aml_interface = AMLInterface()
        env = aml_interface.create_aml_environment(experiment_details["env_name"])

        input_data_path = ""
        output_data_path = ""
        inputs = {
            "input_data": Input(type=AssetTypes.URI_FILE, path=input_data_path)
        }
        outputs = {
            "output_folder": Output(type=AssetTypes.URI_FOLDER, path=output_data_path)
        }

        job = command(
            code=".",  # local path where the code is stored
            command="PYTHONPATH=. python file_to_execute.py --input_data ${{inputs.input_data}} --output_folder ${{outputs.output_folder}}",
            inputs=inputs,
            outputs=outputs,
            environment=env,
            compute=experiment_details["compute_name"],
        )
        submitted_job = aml_interface.submit_command_job(job)

        Parameters
        ----------
        job
            The job to be created or updated.

        Returns
        -------
            The created or updated resource.

        """
        return self.workspace.create_or_update(job)

    def submit_pipeline_job(self, pipeline_job, experiment_name):
        """

        Examples
        --------
        aml_interface = AMLInterface()
        aml_interface.create_aml_environment(
            experiment_details["env_name"]
        )

        input_data_path = ""
        output_data_path = ""
        input_data = Input(type=AssetTypes.URI_FILE, path=input_data_path)

        metadata_pipeline_job = metadata_pipeline(
            input_data=input_data,
            output_data_path=output_data_path
        )
        metadata_pipeline_job.settings.default_compute = experiment_details["compute_name"]

        pipeline_job = aml_interface.submit_pipeline_job(pipeline_job=metadata_pipeline_job,
                                                         experiment_name="metadata_pipeline")
        aml_interface.wait_until_job_completes(pipeline_job.name)

        Parameters
        ----------
        pipeline_job
        experiment_name

        Returns
        -------

        """
        return self.workspace.jobs.create_or_update(
            pipeline_job, experiment_name=experiment_name
        )

    def wait_until_job_completes(self, job_name):
        self.workspace.jobs.stream(job_name)
