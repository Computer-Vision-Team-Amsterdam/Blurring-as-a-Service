import logging

import azureml
import yaml
from azureml.core import (
    ComputeTarget,
    Dataset,
    Datastore,
    Environment,
    Experiment,
    Model,
    ScriptRunConfig,
    Workspace,
)
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.conda_dependencies import CondaDependencies
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.core import Pipeline

# TODO: Adjust how environment is created based on Sebastian's snippet. Try to export packages from poetry and then
#  load them in python: poetry export -f requirements.txt --output requirements.txt

logger = logging.getLogger(__name__)


class AMLInterface:
    """This class provides an interface to interact with Azure ML.

    Attributes
    ----------
    connection_name : str
        Name of the service that the AMLInterface is connected to, e.g. Analyse Services or your own subscription.
    dtap : str
        Environment (development/test/acceptance/production) that the AMLInterface is connected to.
    conn_details : dict
        Dictionary containing various details of the connection, such as subscription ID, resource group, and tenant ID.
    workspace :
        Instance of :class:`azureml.core.Workspace`
    """

    def __init__(self, conn_details_path: str, connection_name: str, dtap: str):
        """Initiate AMLInterface with connection to a specific AML workspace.

        Parameters
        ----------
        conn_details_path : str
            Path where the yaml file with connection details is located - see 'connection_details_example.yml' for an
            example.
        connection_name : str
            Which service to connect to, e.g. Analyse Services or your own subscription. Connection details must be
            present in the connection details file.
        dtap : str
            Which environment (development/test/acceptance/production) to connect to. Connection details must be
            present in the connection details file.
        """
        self.connection_name = connection_name
        self.dtap = dtap
        with open(conn_details_path) as f:
            conn_details = yaml.safe_load(f)
            self.conn_details = conn_details["connections"][self.connection_name][
                self.dtap
            ]

        self.workspace = self._get_workspace()

    def _get_workspace(self) -> azureml.core.Workspace:
        """Get AzureML workspace using connection details of the instance.

        Returns
        -------
        :class:`azureml.core.Workspace`
            Workspace instance
        """
        ia = InteractiveLoginAuthentication(
            tenant_id=self.conn_details["tenant_id"], force=True
        )
        ws = Workspace.get(
            name=self.conn_details["aml_workspace_name"],
            subscription_id=self.conn_details["subscription_id"],
            resource_group=self.conn_details["resource_group"],
            auth=ia,
        )

        logger.info(ws.get_details())
        return ws

    def register_datastore(
        self, datastore_name: str, blob_container: str, account_key: str = ""
    ) -> None:
        """Register a datastore based on a blob storage container.

        Parameters
        ----------
        datastore_name: str
            Name of the datastore to create.
        blob_container: str
            Name of the storage account container that the datastore should link to.
        account_key: str
            Account key for access to the storage account. Can be left empty if authenticating interactively.

        Returns
        -------
            None
        """
        Datastore.register_azure_blob_container(
            workspace=self.workspace,
            datastore_name=datastore_name,
            container_name=blob_container,
            account_name=self.conn_details["storage_account_name"],
            account_key=account_key,
        )

    def register_dataset(
        self, datastore_name: str, file_name: str, dataset_name: str
    ) -> None:
        """Register a dataset based on a file in a datastore. Currently only accepts delimited files which will be
        registered as a tabular dataset.

        Parameters
        ----------
        datastore_name : str
            Name of datastore containing the file to create the dataset from.
        file_name : str
            Name of the file within the datastore to create the dataset from.
        dataset_name : str
            Name of the dataset that will be registered.

        Returns
        -------
            None
        """
        blob_ds = Datastore.get(self.workspace, datastore_name=datastore_name)
        csv_path = [(blob_ds, file_name)]
        tab_ds = Dataset.Tabular.from_delimited_files(path=csv_path)
        tab_ds.register(workspace=self.workspace, name=dataset_name)

    def create_aml_environment(
        self,
        env_name,
        base_dockerfile=None,
        pip_packages=None,
        pip_option=None,
        python_version="3.8.12",
        register=True,
    ):
        """Creates an aml environment that can be of two types:
            - a base environment based on the AzureML-sklearn-0.24-ubuntu18.04-py37-cpu image;
            - a custom environment where the docker image is specified and is possible to install pip packages.

        Parameters
        ----------
        env_name : str
            Name to give to the new environment.
        base_dockerfile : str
            Path where to retrieve the dockerfile to create the custom environment.
        pip_packages : List[str]
            List of packages to install.
        pip_option : str
            Option string, used to add the source of custom packages not present in conda.
            For example: "--extra-index-url https://pkgs.dev.azure.com/CloudCompetenceCenter/Datateam-Sociaal/_packaging/team-AA/pypi/simple/"
        python_version : str
            Version of python to use as default in the environment.
        register : boolean
            Boolean to decide if register the environment.

        Returns
        -------
            None
        """
        if base_dockerfile:
            env = Environment(env_name)
            env.docker.base_image = None
            env.docker.base_dockerfile = base_dockerfile
            # env.python.user_managed_dependencies = managed_dependencies
            # env.python.interpreter_path = '/opt/miniconda/envs/example/bin/python'
            if pip_packages:
                cd = CondaDependencies.create(
                    python_version=python_version, pip_packages=pip_packages
                )
                if pip_option:
                    cd.set_pip_option(pip_option)
                env.python.conda_dependencies = cd

        else:
            original_env = Environment.get(
                workspace=self.workspace,
                name="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu",
            )

            env = original_env.clone(env_name)

        if register:
            self._register_aml_environment(env)

        return env

    def _register_aml_environment(self, environment: azureml.core.Environment) -> None:
        """Register an environment to AzureML.

        Parameters
        ----------
        environment : :class:`azureml.core.Environment`
            Environment instance to be registered.

        Returns
        -------
            None
        """
        environment.register(workspace=self.workspace)

    def get_compute_target(
        self, compute_name: str, vm_size: str = None
    ) -> azureml.core.ComputeTarget:
        """Get or create compute target.

        Parameters
        ----------
        compute_name : str
            Name of the compute to get. Must be an existing compute and must be created through the AzureML UI,
            because there's a bug in the Python SDK that prevents the creation of compute without public IP.
        vm_size : str
            VM size if creating new compute on the fly.  ! Not currently possible

        Returns
        -------
        :class:`azureml.core.ComputeTarget`
            Compute target instance
        """
        try:
            compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
            logger.info("Found existing compute target")
        except ComputeTargetException as e:
            raise Exception(
                "You have to create a compute manually first, because creating one without a public IP "
                "using the python SDK doesn't seem to work. It just ignores the `enable_node_public_ip` setting."
            ) from e
            # logger.info("Creating a new compute target...")
            # compute_config = AmlCompute.provisioning_configuration(
            #     vm_size=vm_size,
            #     min_nodes=0,
            #     max_nodes=1,
            #     idle_seconds_before_scaledown=1800,
            #     subnet_name="DAO-AKS-Workernodes",
            #     vnet_resourcegroup_name=self.conn_details["vnet_resourcegroup_name"],
            #     vnet_name=self.conn_details["vnet_name"],
            #     enable_node_public_ip=False,
            #     identity_type="UserAssigned",
            #     identity_id=[
            #         f"/subscriptions/{self.conn_details['subscription_id']}/resourceGroups/"
            #         f"{self.conn_details['resource_group']}/providers/Microsoft.ManagedIdentity/"
            #         f"userAssignedIdentities/{self.conn_details['user_assigned_identity_id']}"
            #     ]
            # )
            #
            # compute_target = ComputeTarget.create(
            #     self.workspace, compute_name, compute_config
            # )
            # compute_target.wait_for_completion(
            #     show_output=True, timeout_in_minutes=20
            # )
        return compute_target

    def submit_ml_experiment(
        self,
        source_dir: str,
        exp_script: str,
        exp_name: str,
        env_name: str,
        compute_name: str,
        vm_size: str = "STANDARD_D2_V2",
        script_args: list = None,
        wait_for_completion: bool = True,
    ) -> None:
        """Submit an experiment to AzureML.

        Parameters
        ----------
        source_dir : str
            Folder where the experiment script is located.
        exp_script : str
            Path to the script containing the experiment.
        exp_name : str
            Name of the experiment on AML.
        env_name : str
            Name of the environment to use. Must be an existing AML environment.
        compute_name : str
            Name of the compute to use. Must be an existing compute and must be created through the AzureML UI,
            because there's a bug in the Python SDK that prevents the creation of compute without public IP.
        vm_size : str
            VM size if creating new compute on the fly.
        script_args : list
            List of argument names and values to be passed on to the :class:`azureml.core.ScriptRunConfig`.
        wait_for_completion : bool
            If True, function will print experiment output and return after the experiment has finished running. If
            False, function will return immediately after submitting the experiment.

        Returns
        -------
            None
        """
        env = Environment.get(workspace=self.workspace, name=env_name)
        compute_target = self.get_compute_target(compute_name, vm_size)

        if not script_args:
            script_args = []

        script_config = ScriptRunConfig(
            source_directory=source_dir,
            script=exp_script,
            environment=env,
            compute_target=compute_target,
            arguments=script_args,
        )

        experiment = Experiment(workspace=self.workspace, name=exp_name)
        run = experiment.submit(config=script_config)

        if wait_for_completion:
            run.wait_for_completion(show_output=True)

    def create_pipeline(self, steps, exp_name):
        """

        Parameters
        ----------
        steps
        exp_name

        Returns
        -------

        """
        ppl = Pipeline(workspace=self.workspace, steps=steps)
        ppl.validate()

        return Experiment(self.workspace, exp_name).submit(
            ppl, regenerate_outputs=False
        )

    def download_model(self, model_name: str, target_dir: str) -> str:
        """Download a registered model from AzureML.

        Parameters
        ----------
        model_name : str
            Name of the model to download.
        target_dir : str
            Folder to save downloaded file(s) in. Defaults to ".".

        Returns
        -------
        str
            Path to the downloaded file(s)
        """
        model = Model(self.workspace, model_name)
        return model.download(target_dir)
