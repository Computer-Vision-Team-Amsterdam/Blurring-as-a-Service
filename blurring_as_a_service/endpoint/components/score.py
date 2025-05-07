import base64
import io
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from yolo_model_development_kit.inference_pipeline.source.model_result import (
    ModelResult,
)
from yolo_model_development_kit.inference_pipeline.source.output_image import (
    OutputImage,
)

print("Current working directory:", os.getcwd())
print("Contents:", os.listdir(os.getcwd()))

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
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
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "yolov8m_1280_v2.2_curious_hill_12.pt"
    )
    print("Init started")
    model = YOLO(
        model=model_path,
        task="detect",
    )
    print("Init complete")


def run(raw_data):
    try:
        data = json.loads(raw_data)
        user_id = data.get("user_id", "unknown")
        print(f"Request received from user: {user_id}")

        image_data = data.get("data")
        if image_data is None:
            return json.dumps({"error": "No image data provided."})

        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        results = model(image)
        result = results[0].cpu()

        inference_settings = settings["inference_pipeline"]
        target_classes = inference_settings["target_classes"]
        sensitive_classes = inference_settings["sensitive_classes"]
        inference_params = {
            "imgsz": inference_settings["model_params"].get("img_size", 640),
            "save": inference_settings["model_params"].get("save_img_flag", False),
            "save_txt": inference_settings["model_params"].get("save_txt_flag", False),
            "save_conf": inference_settings["model_params"].get(
                "save_conf_flag", False
            ),
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
        model_result = ModelResult(
            model_result=result,
            target_classes=target_classes,
            sensitive_classes=sensitive_classes,
            target_classes_conf=target_classes_conf,
            sensitive_classes_conf=sensitive_classes_conf,
            save_image=False,
            save_labels=False,
            save_all_images=False,
        )
        output_image = OutputImage(result.orig_img.copy())
        model_result.calculate_bounding_boxes()

        if len(model_result.sensitive_bounding_boxes):
            output_image.blur_inside_boxes(boxes=model_result.sensitive_bounding_boxes)
        else:
            print("No sensitive classes detected, skipping blurring.")

        success, encoded_image = cv2.imencode(".jpg", output_image.image)
        if not success:
            print("Image encoding failed.")
            return json.dumps({"error": "Image encoding failed."})

        annotated_image_b64 = base64.b64encode(encoded_image).decode("utf-8")
        metadata = {
            "persons_count": int((model_result.boxes.cls == 0).sum()),
            "licence_plates_count": int((model_result.boxes.cls == 1).sum()),
        }
        print(f"Metadata: {metadata}")
        return json.dumps(
            {"annotated_image": annotated_image_b64, "metadata": metadata}
        )
    except Exception as e:
        print(f"Error processing request: {e}")
        return json.dumps({"error": str(e)})
