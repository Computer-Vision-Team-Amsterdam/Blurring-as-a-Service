from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.entities import (
    CodeConfiguration,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
)

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

# DO NOT import relative paths before setting up the logger.
# Exception, of course, is settings to set up the logger.
BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"], __name__)
azureLoggingConfigurer.setup_baas_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402

aml_interface = AMLInterface()


def main():
    ml_client = aml_interface.ml_client
    endpoint_name = "endpt-cvt-baas"
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name, auth_mode="key"  # you can also use token if preferred
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=f"{endpoint_name}-deployment",
        endpoint_name=endpoint.name,
        model="OOR-model:2",
        environment="baas-environment:108",
        code_configuration=CodeConfiguration(
            code="blurring_as_a_service/endpoint/components/",  # directory containing your score.py file
            scoring_script="scoring.py",
        ),
        instance_type="Standard_DS3_v2",  # choose an instance type that fits your workload
        instance_count=1,
    )

    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Update the endpoint traffic to route 100% to this deployment.
    ml_client.online_endpoints.begin_update(
        endpoint, traffic={"yolov11-deployment": 100}
    ).result()


if __name__ == "__main__":
    main()
