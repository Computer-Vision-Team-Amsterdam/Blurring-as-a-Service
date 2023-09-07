import logging

from opencensus.ext.azure.log_exporter import AzureLogHandler


def setup_azure_logging_from_config(cfg):
    logging.basicConfig(**cfg["basic_config"])
    instrumentation_key = (
        "InstrumentationKey=179725b3-3e51-4191-9070-d7449d34420b;IngestionEndpoint=https"
        "://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope"
        ".livediagnostics.monitor.azure.com/ "
    )
    azure_log_handler = AzureLogHandler(connection_string=instrumentation_key)
    for pkg in cfg["own_packages"]:
        logging.getLogger(pkg).setLevel(cfg["loglevel_own"])
        logging.getLogger(pkg).addHandler(azure_log_handler)
