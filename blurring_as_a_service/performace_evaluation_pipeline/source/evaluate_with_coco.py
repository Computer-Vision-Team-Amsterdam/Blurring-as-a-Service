import json
import warnings

from pycocotools.coco import COCO

from blurring_as_a_service.performace_evaluation_pipeline.metrics.custom_coco_evaluator import (
    CustomCOCOeval,
)


def coco_evaluation(
    coco_annotations_json: str, coco_predictions_json: str, metrics_metadata: dict
):
    """
    Runs COCO evaluation on the output of YOLO validation

    Parameters
    ----------
    coco_annotations_json: annotations in the COCO format compatible with yolov5. Comes from the metadata pipeline
    coco_predictions_json: predictions in COCO format of the yolov5 run.
    metrics_metadata: info about image sizes and areas for sanity checks.

    Returns
    -------

    """

    COCO_gt = COCO(coco_annotations_json)  # init annotations api
    try:
        COCO_dt = COCO_gt.loadRes(coco_predictions_json)  # init predictions api
    except FileNotFoundError:
        raise Exception(
            f"The specified file '{coco_predictions_json}' was not found."
            f"The file is created at the above path if you run yolo validation with"
            f"the --save-json flag enabled."
        )
    evaluation = CustomCOCOeval(COCO_gt, COCO_dt, "bbox")

    # Opening JSON file
    with open(coco_annotations_json) as f:
        data = json.load(f)

    height = data["images"][0]["height"]
    width = data["images"][0]["width"]
    if (
        height != metrics_metadata["image_height"]
        or width != metrics_metadata["image_width"]
    ):
        warnings.warn(
            f"You're trying to run evaluation on images of size {height} x {width}, "
            "but the coco annotations have been generated from images of size "
            f"{metrics_metadata['image_height']} x {metrics_metadata['image_width']}."
            "Why is it a problem? Because the coco annotations that the metadata produces and the "
            " *_predictions.json produced by the yolo run are both in absolute format,"
            "so we must compare use the same image sizes."
            "Solutions: 1. Use images for validation that are the same size as the ones you used in the "
            "data labeling project. 2. After you export the json from the labeling project, overwrite"
            "the heights and widths from the images, since that version is still using normalized values."
        )

    image_names = [image["id"] for image in data["images"]]
    evaluation.params.imgIds = image_names  # image IDs to evaluate
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
