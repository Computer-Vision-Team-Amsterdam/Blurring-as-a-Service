import json

from pycocotools.coco import COCO

from blurring_as_a_service.performace_evaluation_pipeline.metrics.custom_coco_evaluator import (
    CustomCOCOeval,
)


def coco_evaluation(annotations_json, yolo_output_folder):
    predictions_json = (
        f"{yolo_output_folder}/exp/last-purple_boot_3l6p24vb_predictions.json"
    )
    annotations_json = COCO(annotations_json)  # init annotations api
    predictions = annotations_json.loadRes(predictions_json)  # init predictions api
    evaluation = CustomCOCOeval(annotations_json, predictions, "bbox")

    # Opening JSON file
    with open(annotations_json) as f:
        data = json.load(f)

    image_names = [image["id"] for image in data["images"]]
    evaluation.params.imgIds = image_names  # image IDs to evaluate
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
