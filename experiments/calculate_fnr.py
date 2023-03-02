from blurring_as_a_service.metrics.custom_metrics_calculator import (
    CustomMetricsCalculator,
)


def run_custom_metrics_calculator_1mar():
    yolo_run_path = "../yolov5/runs/val/exp13/"
    source = f"{yolo_run_path}/labels_tagged"
    coco_file_with_categories = (
        "exp_with_tagged_validation_1mar/in:coco-format/validation-tagged.json"
    )
    markdown_output_path = (
        f"{yolo_run_path}/custom_metrics_result_imgsz_640_with_tba.md"
    )

    metrics_calculator = CustomMetricsCalculator(source, coco_file_with_categories)
    metrics_calculator.calculate_and_store_metrics(markdown_output_path)
    # metrics_calculator._plot_area_distribution(path=".", plot_name="GT_bboxes_2")


def run_custom_metrics_calculator_21feb():
    yolo_run_path = "../yolov5/runs/val/exp9_baseline/"
    source = f"{yolo_run_path}/labels_tagged"
    coco_file_with_categories = (
        "exp_with_tagged_validation_21feb/in:coco-format/validation-tagged.json"
    )
    markdown_output_path = (
        f"{yolo_run_path}/custom_metrics_result_imgsz_640_with_tba.md"
    )

    metrics_calculator = CustomMetricsCalculator(source, coco_file_with_categories)
    metrics_calculator.calculate_and_store_metrics(markdown_output_path)
    # metrics_calculator._plot_area_distribution(path=".", plot_name="GT_bboxes_2")
