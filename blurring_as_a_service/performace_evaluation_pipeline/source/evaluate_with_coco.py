import json

from pycocotools.coco import COCO

from blurring_as_a_service.metrics.custom_coco_evaluator import CustomCOCOeval


def coco_evaluation(annotations_json: str, yolo_output_folder: str, yolo_run_name: str):
    """
    Runs COCO evaluation on the output of YOLO validation

    Parameters
    ----------
    annotations_json: annotations in the COCO format compatible with yolov5. Comes from azure_coco_to_coco_converter.py
    yolo_output_folder: output path of yolo validation
    experiment_name: yolo run name

    Returns
    -------

    """
    predictions_json = f"{yolo_output_folder}/{yolo_run_name}/last-purple_boot_3l6p24vb_predictions.json"
    coco_annotations = COCO(annotations_json)  # init annotations api
    predictions = coco_annotations.loadRes(predictions_json)  # init predictions api
    evaluation = CustomCOCOeval(coco_annotations, predictions, "bbox")

    # Opening JSON file
    with open(annotations_json) as f:
        data = json.load(f)

    image_names = [image["id"] for image in data["images"]]
    evaluation.params.imgIds = image_names  # image IDs to evaluate
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
