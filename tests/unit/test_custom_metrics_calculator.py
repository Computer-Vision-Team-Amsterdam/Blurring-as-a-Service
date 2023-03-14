from blurring_as_a_service.performace_evaluation_pipeline.metrics.custom_metrics_calculator import (
    CustomMetricsCalculator,
)


def expected_output():
    """
    Check if the markdown file has been written correctly.

    Returns
    -------
    list
        Expected YOLO conversion.
    """
    return "| Category | Value | True Positives | False Negatives | False Negative Rate |\n"


def test_custom_metrics_calculator():
    source = "../../local_test_data/in:coco-format/labels_tagged"
    coco_file_with_categories = (
        "../../local_test_data/in:coco-format/blur_v0.1/validation-tagged.json"
    )
    markdown_output_path = (
        "../../local_test_data/out:yolo-format/custom_metrics_result.md"
    )

    metrics_calculator = CustomMetricsCalculator(source, coco_file_with_categories)
    metrics_calculator.calculate_and_store_metrics(markdown_output_path)

    with open(markdown_output_path, "r") as f:
        assert f.readline() == expected_output()
