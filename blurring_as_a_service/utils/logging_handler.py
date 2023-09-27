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
    instrumentation_key = logging_settings["ai_instrumentation_key"]
    azure_log_handler = AzureLogHandler(connection_string=instrumentation_key)
    for pkg in logging_settings["own_packages"]:
        logging.getLogger(pkg).setLevel(logging_settings["loglevel_own"])
        logging.getLogger(pkg).addHandler(azure_log_handler)

    logger = logging.getLogger("__main__")
    return logger
