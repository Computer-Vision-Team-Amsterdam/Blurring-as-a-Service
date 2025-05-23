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


def main():
    aml_interface = AMLInterface()
    ml_client = aml_interface.ml_client
    endpoint_name = settings["api_endpoint"]["endpoint_name"]
    deployment_color = settings["api_endpoint"]["deployment_color"]

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="aml_token",
        public_network_access="Disabled",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    deployment = ManagedOnlineDeployment(
        name=f"{endpoint_name}-{deployment_color}",
        endpoint_name=endpoint.name,
        model=f"{settings['api_endpoint']['model_name']}:{settings['api_endpoint']['model_version']}",
        environment=f"{settings['aml_experiment_details']['env_name']}:{settings['aml_experiment_details']['env_version']}",
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="blurring_as_a_service/endpoint/components/score.py",
        ),
        instance_type=settings["api_endpoint"]["instance_type"],
        instance_count=1,
        egress_public_network_access="disabled",
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()


if __name__ == "__main__":
    main()
