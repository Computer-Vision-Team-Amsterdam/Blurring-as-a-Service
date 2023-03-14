import argparse
import sys
from pathlib import Path

from blurring_as_a_service.performace_evaluation_pipeline.metrics.custom_metrics_calculator import (
    CustomMetricsCalculator,
)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # experiments root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def main(opt):
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    metrics_calculator = CustomMetricsCalculator(
        tagged_validation_folder=opt.labels_tagged,
        coco_file_with_categories=opt.coco_tagged_annotations,
    )
    metrics_calculator.calculate_and_store_metrics(
        markdown_output_path=save_dir / "custom_metrics_result.md"
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels_tagged",
        help="path to folder with tagged labels from yolo tagged validation",
    )
    parser.add_argument(
        "--coco-tagged-annotations",
        help="Azure Labelling COCO json file with tagged annotations",
    )
    parser.add_argument("--project", default=ROOT / "runs", help="save to project/name")
    parser.add_argument(
        "--name", default="exp_{fill_in_date_here}", help="save to project/name"
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    # note: yolo/runs/exp13 corresponds to 1st of March, yolo/runs/val9_baseline to 21 of Feb
    opt.labels_tagged = "yolov5/runs/val/exp9_baseline/labels_tagged"
    opt.coco_tagged_annotations = "experiments/exp_with_tagged_validation_21feb/in:coco-format/validation-tagged.json"

    main(opt)
