from cvtoolkit.database.database_handler import DBConfigSQLAlchemy

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

settings = BlurringAsAServiceSettings.get_settings()


def create_db_connector() -> DBConfigSQLAlchemy:
    """
    Creates a database connector configuration object.

    This function retrieves database parameters from the settings and validates
    that all required parameters are present. If any required parameter is missing
    or incomplete, it raises a ValueError. Otherwise, it creates and returns a
    DBConfigSQLAlchemy object with the provided database parameters.

    Returns
    -------
    DBConfigSQLAlchemy
        A configuration object for connecting to the database.

    Raises
    ------
    ValueError
        If any of the required database parameters are missing or incomplete.

    """
    required_keys = ["db_username", "db_name", "db_hostname", "client_id"]
    database_parameters = settings["inference_pipeline"]["database_parameters"]

    missing_keys = [key for key in required_keys if not database_parameters.get(key)]
    if missing_keys:
        raise ValueError(
            f"Database credentials are missing or incomplete: {', '.join(missing_keys)}"
        )

    db_config = DBConfigSQLAlchemy(
        database_parameters["db_username"],
        database_parameters["db_hostname"],
        database_parameters["db_name"],
        database_parameters["client_id"],
    )
    return db_config
