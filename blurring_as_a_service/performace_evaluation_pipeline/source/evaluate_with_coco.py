import json

from pycocotools.coco import COCO

from metrics.customCocoEval import CustomCOCOeval


def coco_evaluation(annotations_json, yolo_output_folder):

    predictions_json = (
        f"{yolo_output_folder}/exp/last-purple_boot_3l6p24vb_predictions.json"
    )
    anno = COCO(annotations_json)  # init annotations api
    pred = anno.loadRes(predictions_json)  # init predictions api
    eval = CustomCOCOeval(anno, pred, "bbox")

    # Opening JSON file
    f = open(annotations_json)
    data = json.load(f)

    image_names = [image["id"] for image in data["images"]]
    eval.params.imgIds = image_names  # image IDs to evaluate
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
