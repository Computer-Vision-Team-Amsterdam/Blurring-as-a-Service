import os

from blurring_as_a_service.utils.logging_handler import (  # noqa: E402
    setup_azure_logging_from_config,
)

logger = setup_azure_logging_from_config()


class LockFile:
    def __init__(self, file_path):
        self.file_path = file_path
        self.locked_file_path = self.file_path + ".lock"

    def __enter__(self):
        try:
            os.rename(self.file_path, self.locked_file_path)

            with open(self.locked_file_path, "r") as src:
                return src
        except FileNotFoundError as e:
            logger.info(f"File {self.locked_file_path} not found: {e}")
        except Exception as e:
            logger.error(f"Error occurred while reading {self.locked_file_path}: {e}")

    def __exit__(self, exc_type, exc_value, traceback):
        os.rename(self.locked_file_path, self.file_path)
        if exc_type is not None:
            logger.error(f"An exception occurred within the 'with' block: {exc_type}, {exc_value}")