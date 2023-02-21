import glob
import os
import sys

from mldesigner import Input, command_component

sys.path.append("../../..")

from blurring_as_a_service.metrics.metrics_utils import (  # noqa: E402
    process_image_labels,
)
from blurring_as_a_service.metrics.total_blurred_area import (  # noqa: E402
    TotalBlurredArea,
)
from blurring_as_a_service.metrics.upper_blurred_area import (  # noqa: E402
    UpperBlurredArea,
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
)
def evaluate_with_cvt_metrics(
    mounted_dataset: Input(type="uri_folder"), yolo_output_folder: Input(type="uri_folder")  # type: ignore # noqa: F821
):
    true_labels = [file for file in glob.glob(f"{mounted_dataset}/labels/val/*.txt")]
    pred_labels = [file for file in glob.glob(f"{yolo_output_folder}/exp/labels/*.txt")]
    print(true_labels)
    print(pred_labels)
    inputs = [
        {"true": true_label, "predicted": pred_label}
        for true_label, pred_label in zip(true_labels, pred_labels)
    ]

    tba = TotalBlurredArea()
    uba = UpperBlurredArea()

    for i, input in enumerate(inputs):
        tba_true, tba_pred, uba_true, uba_pred = process_image_labels(input)
        print(i)
        tba.add_mask(tba_true, tba_pred)
        uba.add_mask(uba_true, uba_pred)

    tba.summary("Statistics for Total Blurred Area:")
    uba.summary("Statistics for Upper Blurred Area:")
