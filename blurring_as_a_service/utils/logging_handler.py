import logging

from opencensus.ext.azure.log_exporter import AzureLogHandler

# class CustomLogger:
#     def __init__(self, cfg):
#         self._logger = None  # Initialize the logger attribute to None
#         self.cfg = cfg
#         self.setup_azure_logging_from_config()
#
#     def setup_azure_logging(self):
#         self._logger = logging.getLogger('opencensus')
#         self._logger.setLevel(logging.WARNING)
#         # logger.addHandler(logging.StreamHandler())
#         instrumentation_key = "InstrumentationKey=179725b3-3e51-4191-9070-d7449d34420b;IngestionEndpoint=https" \
#                               "://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope" \
#                               ".livediagnostics.monitor.azure.com/ "
#         self._logger.handlers.clear()
#         self._logger.addHandler(AzureLogHandler(connection_string=instrumentation_key))


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
