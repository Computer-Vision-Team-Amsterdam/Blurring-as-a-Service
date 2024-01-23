import logging
import os

logger = logging.getLogger(__name__)


class LockFile:
    """
    Locks a file in case of multiprocess/multithread to avoid two processes accessing the same file.

    Examples
    --------
    with LockFile(file_path) as locked_file:
        for line in locked_file:
            print(line)
    """

    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path
            Path of the file to be locked.
        """
        self.file_path = file_path
        self.locked_file_path = self.file_path + ".lock"

    def __enter__(self):
        """
        Entering the with clause the file is renamed adding .lock postfix to avoid access.

        Returns
        -------
        Locked file object.
        """
        try:
            os.rename(self.file_path, self.locked_file_path)
            src = open(self.locked_file_path, "r")
            return src
        except FileNotFoundError as e:
            logger.error(f"File {self.locked_file_path} not found: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error occurred while reading {self.locked_file_path}: {e}")
            raise e

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exiting the with clause the file is renamed back to the original name.
        """
        try:
            os.rename(self.locked_file_path, self.file_path)
        except Exception as e:
            logging.error(f"Error occurred while unlocking file {self.file_path}: {e}")

        if exc_type is not None:
            logging.error(
                f"An exception occurred within the 'with' block: {exc_type}, {exc_value}"
            )
