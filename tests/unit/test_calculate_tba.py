from blurring_as_a_service.metrics.tba_calculator import (
    collect_and_store_tba_results_per_class_and_size,
)


def expected_output():
    """
    Check if the markdown file has been written correctly.

    Returns
    -------
    list
        Expected structure for markdown.
    """
    return (
        "| Person Small | Person Medium | Person Large "
        "| License Plate Small | License Plate Medium | License Plate Large | \n"
    )


def test_custom_metrics_calculator():
    true_path = "../../local_test_data/sample-tagged/labels/val"
    pred_path = "../../yolov5/runs/val/exp23/labels"
    markdown_output_path = "tba_results.md"

    collect_and_store_tba_results_per_class_and_size(
        true_path, pred_path, markdown_output_path=markdown_output_path
    )
