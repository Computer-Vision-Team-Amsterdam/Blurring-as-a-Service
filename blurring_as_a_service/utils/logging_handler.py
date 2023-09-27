import logging
import os

from opencensus.ext.azure.log_exporter import AzureLogHandler

from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)


def setup_azure_logging_from_config() -> logging.Logger:
    logging_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)["logging"]

    logging.basicConfig(**logging_settings["basic_config"])
    instrumentation_key = (
        "InstrumentationKey=179725b3-3e51-4191-9070-d7449d34420b;IngestionEndpoint=https"
        "://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope"
        ".livediagnostics.monitor.azure.com/ "
    )
    azure_log_handler = AzureLogHandler(connection_string=instrumentation_key)
    for pkg in logging_settings["own_packages"]:
        logging.getLogger(pkg).setLevel(logging_settings["loglevel_own"])
        logging.getLogger(pkg).addHandler(azure_log_handler)

    logger = logging.getLogger("__main__")
    return logger
