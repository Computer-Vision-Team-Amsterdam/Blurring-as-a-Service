import os
import sys
from pathlib import Path

from mldesigner import Input, command_component  # noqa: E402

sys.path.append("../../..")

from blurring_as_a_service.performace_evaluation_pipeline.metrics.fnr_calculator import (  # noqa: E402
    FalseNegativeRateCalculator,
)
from blurring_as_a_service.performace_evaluation_pipeline.metrics.tba_calculator import (  # noqa: E402
    collect_and_store_tba_results_per_class_and_size,
)
from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
aml_experiment_settings = BlurringAsAServiceSettings.set_from_yaml(config_path)[
    "aml_experiment_details"
]


@command_component(
    name="evaluate_with_cvt_metrics",
    display_name="Evaluation with CVT metrics",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def evaluate_with_cvt_metrics(
    mounted_dataset: Input(type="uri_folder"),  # type: ignore # noqa: F821
    yolo_output_folder: Input(type="uri_folder"),  # type: ignore # noqa: F821
    annotations_for_custom_metrics: Input(type="uri_file"),  # type: ignore # noqa: F821
    yolo_run_name: str,  # type: ignore # noqa: F821
):
    true_path = f"{mounted_dataset}/labels/val"
    pred_path = f"{yolo_output_folder}/{yolo_run_name}/labels"
    pred_path_tagged = f"{yolo_output_folder}/{yolo_run_name}/labels_tagged"

    # ======== Total Blurred Area metric ========= #
    collect_and_store_tba_results_per_class_and_size(
        true_path, pred_path, markdown_output_path="tba_results.md"
    )

    # ======== False Negative Rate metric ========= #

    if Path.exists(Path(pred_path_tagged)):
        metrics_calculator = FalseNegativeRateCalculator(
            pred_path_tagged, annotations_for_custom_metrics
        )
        metrics_calculator.calculate_and_store_metrics(
            markdown_output_path="fnr_results.md"
        )
    else:
        raise FileNotFoundError(
            "False Negative Rate metrics can only be run with tagged validation. "
            "labels_tagged folder not found. Make sure you run val.py with the --tagged-data "
            "flag in case you want to compute this metric."
        )
