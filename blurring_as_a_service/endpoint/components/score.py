import base64
import io
import json
import logging
import os
import sys

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from yolo_model_development_kit.inference_pipeline.source.model_result import (
    ModelResult,
)
from yolo_model_development_kit.inference_pipeline.source.output_image import (
    OutputImage,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from blurring_as_a_service.settings.settings import (  # noqa: E402
    BlurringAsAServiceSettings,
)

settings = BlurringAsAServiceSettings.set_from_yaml(config_path)
azureLoggingConfigurer = AzureLoggingConfigurer(settings["logging"])
azureLoggingConfigurer.setup_baas_logging()
logger = logging.getLogger("api_endpoint")


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global settings
    global logger

    logger.info("Init started")
    try:
        # AZUREML_MODEL_DIR is an environment variable created during deployment.
        # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
        # Please provide your model's folder name if there is one
        model_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR"), "yolov8m_1280_v2.2_curious_hill_12.pt"
        )
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if settings is None:
            raise RuntimeError("Configuration settings could not be loaded.")

        logger.info(f"Loading model from: {model_path}")
        model = YOLO(
            model=model_path,
            task="detect",
        )
        logger.info("Init complete")
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: Model file not found. {e}")
        raise
    except EnvironmentError as e:
        logger.error(f"Initialization failed: Environment configuration error. {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Initialization failed: Critical runtime error. {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during init: {e}", exc_info=True)
        raise


def run(raw_data):
    """
    This function is called for every invocation of the endpoint.
    Parameters:
        raw_data (str): The raw request data in JSON format. The JSON contains:
            - "data" field containing the image encoded in base64.
            - "user_id" field containing the email of who is using the API.
        }
    Returns:
        tuple: A tuple containing the response (str) and HTTP status code (int).
    """
    global model
    global settings
    global logger

    if model is None:
        logger.error(
            "Model is not initialized. This should not happen if init succeeded."
        )
        error_response = json.dumps(
            {"error": "Model not initialized. Service is in a failed state."}
        )
        return error_response, 503

    if settings is None:
        logger.error(
            "Configuration settings are not loaded. This should not happen if init succeeded."
        )
        error_response = json.dumps(
            {"error": "Service configuration not loaded. Service is in a failed state."}
        )
        return error_response, 503

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON received: {e}", exc_info=True)
        error_response = json.dumps(
            {"error": "Invalid JSON format.", "details": str(e)}
        )
        return error_response, 400

    try:
        user_id = data.get("user_id", "unknown")
        logger.info(f"Request received from user: {user_id}")

        image_data = data.get("data")
        if image_data is None:
            logger.warning(
                "No 'data' field (base64 image string) provided in the request."
            )
            error_response = json.dumps({"error": "Missing 'data' field for image."})
            return error_response, 400

        try:
            image_bytes = base64.b64decode(image_data)
        except base64.binascii.Error as e:
            logger.error(f"Invalid base64 data: {e}", exc_info=True)
            error_response = json.dumps(
                {"error": "Invalid base64 encoded image data.", "details": str(e)}
            )
            return error_response, 400

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        except UnidentifiedImageError as e:
            logger.error(f"Cannot identify image file: {e}", exc_info=True)
            error_response = json.dumps(
                {"error": "Invalid or unsupported image format.", "details": str(e)}
            )
            return error_response, 400
        except IOError as e:
            logger.error(f"IOError opening image: {e}", exc_info=True)
            error_response = json.dumps(
                {"error": "Could not read image data.", "details": str(e)}
            )
            return error_response, 400
        except Exception as e:
            logger.error(f"Error processing image input: {e}", exc_info=True)
            error_response = json.dumps(
                {"error": "Failed to process input image.", "details": str(e)}
            )
            return error_response, 422

        try:
            results = model(image)
            if not results or not results[0]:
                logger.error("Model inference returned empty results.")
                error_response = json.dumps(
                    {"error": "Model inference failed to produce results."}
                )
                return error_response, 500
            result = results[0].cpu()
        except RuntimeError as e:
            logger.error(f"Runtime error during model inference: {e}", exc_info=True)
            error_response = json.dumps(
                {"error": "Model inference failed.", "details": str(e)}
            )
            return error_response, 500
        except Exception as e:
            logger.error(f"Unexpected error during model inference: {e}", exc_info=True)
            error_response = json.dumps(
                {
                    "error": "An unexpected error occurred during model processing.",
                    "details": str(e),
                }
            )
            return error_response, 500

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
            logger.info("No sensitive classes detected, skipping blurring.")

        try:
            success, encoded_image = cv2.imencode(".jpg", output_image.image)
            if not success:
                logger.error("Image encoding to JPG failed.")
                error_response = json.dumps(
                    {"error": "Failed to encode processed image."}
                )
                return error_response, 500
        except cv2.error as e:
            logger.error(f"OpenCV error during image encoding: {e}", exc_info=True)
            error_response = json.dumps(
                {
                    "error": "Failed to encode processed image due to OpenCV error.",
                    "details": str(e),
                }
            )
            return error_response, 500
        except Exception as e:
            logger.error(f"Unexpected error during image encoding: {e}", exc_info=True)
            error_response = json.dumps(
                {
                    "error": "An unexpected error occurred during image encoding.",
                    "details": str(e),
                }
            )
            return error_response, 500

        annotated_image_b64 = base64.b64encode(encoded_image).decode("utf-8")
        metadata = {
            "persons_count": int((model_result.boxes.cls == 0).sum()),
            "licence_plates_count": int((model_result.boxes.cls == 1).sum()),
        }
        logger.info(f"Processing successful for user: {user_id}. Metadata: {metadata}")

        response_payload = json.dumps(
            {"annotated_image": annotated_image_b64, "metadata": metadata}
        )
        return response_payload, 200
    except cv2.error as e:
        logger.error(f"An OpenCV error occurred: {e}", exc_info=True)
        error_response = json.dumps(
            {"error": "An error occurred during image processing.", "details": str(e)}
        )
        return error_response, 500
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in run function: {e}", exc_info=True
        )
        error_response = json.dumps(
            {
                "error": "An unexpected internal server error occurred.",
                "details": str(e),
            }
        )
        return error_response, 500
