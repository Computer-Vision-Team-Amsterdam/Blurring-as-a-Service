import base64
import io
import json
import os
import secrets
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from yolo_model_development_kit.inference_pipeline.source.output_image import (
    OutputImage,
)

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

settings = BlurringAsAServiceSettings.set_from_yaml(config_path)


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    # model_path = os.path.join(
    #     os.getenv("AZUREML_MODEL_DIR"), "model/yolov8m_1280_v2.2_curious_hill_12.pt"
    # )
    print("Init started")
    model = YOLO(
        model="blurring_as_a_service/endpoint/model/yolov8m_1280_v2.2_curious_hill_12.pt",
        task="detect",
    )
    print("Init complete")


def run(raw_data):
    # try:
    # Assume the incoming data is JSON with a key "data" containing the base64 image string.
    data = json.loads(raw_data)
    image_data = data.get("data")
    if image_data is None:
        return json.dumps({"error": "No image data provided."})

    # Decode the base64 image.
    image_bytes = base64.b64decode(image_data)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run the model (adjust this call based on your model's API).
    results = model(image)

    result = results[0]
    # print(f"result cpu: {result}")
    boxes = result.boxes.numpy()
    inference_settings = settings["inference_pipeline"]
    target_classes = inference_settings["target_classes"]
    sensitive_classes = inference_settings["sensitive_classes"]
    inference_params = {
        "imgsz": inference_settings["model_params"].get("img_size", 640),
        "save": inference_settings["model_params"].get("save_img_flag", False),
        "save_txt": inference_settings["model_params"].get("save_txt_flag", False),
        "save_conf": inference_settings["model_params"].get("save_conf_flag", False),
        "conf": inference_settings["model_params"].get("conf", 0.25),
        "project": "inference",
    }
    target_classes_conf = (
        inference_settings["target_classes_conf"]
        if inference_settings["target_classes_conf"]
        else inference_params["conf"]
    )
    sensitive_classes_conf = (
        inference_settings["sensitive_classes_conf"]
        if inference_settings["sensitive_classes_conf"]
        else inference_params["conf"]
    )
    output_image = OutputImage(result.orig_img.copy())

    target_idxs = np.where(
        np.in1d(boxes.cls, target_classes) & (boxes.conf >= target_classes_conf)
    )[0]

    sensitive_idxs = np.where(
        np.in1d(boxes.cls, sensitive_classes) & (boxes.conf >= sensitive_classes_conf)
    )[0]
    if len(sensitive_idxs) > 0:
        sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
        output_image.blur_inside_boxes(boxes=sensitive_bounding_boxes)

    if len(target_idxs) > 0:
        target_bounding_boxes = boxes[target_idxs].xyxy
        target_categories = [int(box.cls) for box in boxes[target_idxs]]
        category_colors = defaultdict(
            lambda: (
                secrets.randbelow(256),
                secrets.randbelow(256),
                secrets.randbelow(256),
            ),
            OutputImage.DEFAULT_COLORS,
        )
        output_image.draw_bounding_boxes(
            boxes=target_bounding_boxes,
            categories=target_categories,
            colour_map=category_colors,
        )

    #     # if save_image or save_all_images:
    #     #     _save_image(output_folder, image_file_name)
    #     # if save_labels and len(target_idxs) > 0:
    #     #     annotation_str = _get_annotation_string_from_boxes(
    #     #         boxes[target_idxs]
    #     #     )
    #     #     _save_labels(annotation_str, output_folder, image_file_name)

    # Encode the annotated image as JPEG.
    success, encoded_image = cv2.imencode(".jpg", output_image.image)
    if not success:
        print("Image encoding failed.")
        return json.dumps({"error": "Image encoding failed."})

    # Return the annotated image as a base64-encoded string.
    annotated_image_b64 = base64.b64encode(encoded_image).decode("utf-8")
    return json.dumps({"annotated_image": annotated_image_b64})
    # except Exception as e:
    #     return json.dumps({"error": str(e)})
