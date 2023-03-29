from blurring_as_a_service.metrics.tba_calculator import (
    collect_and_store_tba_results_per_class_and_size,
)

true_path = "local_test_data/validation-tagged/labels/val"
pred_path = "yolov5/runs/val/exp9_baseline/labels"

collect_and_store_tba_results_per_class_and_size(
    true_path, pred_path, markdown_output_path="tba_results.md"
)
