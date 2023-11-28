import os
import subprocess
import sys
from pathlib import Path

from mldesigner import command_component

sys.path.append("../../..")
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
aml_experiment_settings = settings["aml_experiment_details"]

# Construct the path to the yolov5 package
yolov5_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolov5")
)


def run_cmd(cmd, cwd: Path = "./") -> int:
    """Run the command and returns the result."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
    )
    if result.returncode != 0:
        print(f"Failed with error {result.stdout}.")
    else:
        print(f"Successfully executed! Output: \n{result.stdout}")
    return result.returncode

@command_component(
    name="move_files",
    display_name="Move files",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def move_files(execution_time: str):
    """
    Pipeline step to detect the areas to blur.

    Parameters
    ----------
    execution_time:
        Datetime containing when the job was executed. Used to name the folder.

    """
    src_uri = f"https://{TODO}.blob.core.windows.net/{TODO}-input/*"
    dest_uri = f"https://{TODO}.blob.core.windows.net/{TODO}-input-structured/{execution_time}/"

    # TODO what if we need to do azcopy login?
    download_cmd = f"azcopy copy '{src_uri}' '{dest_uri}' " \
                   "--recursive --skip-version-check --output-level essential"

    result = run_cmd(download_cmd)
    if result:
        print(f"Failed to download model files with URL: {src_uri}")
        return False