import os
import sys

from mldesigner import Input, command_component  # noqa: E402

sys.path.append("../../..")


from blurring_as_a_service.metrics.custom_metrics_calculator import (  # noqa: E402
    CustomMetricsCalculator,
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
    experiment_name: str,  # type: ignore # noqa: F821
):
    true_path = f"{mounted_dataset}/labels/val"
    pred_path = f"{yolo_output_folder}/{experiment_name}/labels"

    # ======== Total Blurred Area metric ========= #
    collect_and_store_tba_results_per_class_and_size(
        true_path, pred_path, markdown_output_path="tba_results.md"
    )

    # ======== False Negative Rate metric ========= #

    metrics_calculator = CustomMetricsCalculator(
        true_path, annotations_for_custom_metrics
    )
    metrics_calculator.calculate_and_store_metrics(
        markdown_output_path="fnr_results.md"
    )
