import logging
from pathlib import Path
from typing import Dict, List

from cvtoolkit.database.baas_tables import DetectionInformation, ImageProcessingStatus
from sqlalchemy.exc import SQLAlchemyError
from ultralytics.engine.results import Results
from yolo_model_development_kit.inference_pipeline.source.YOLO_inference import (
    YOLOInference,
)

from blurring_as_a_service.inference_pipeline.source.db_utils import create_db_connector

logger = logging.getLogger("inference_pipeline")


class BaaSInference(YOLOInference):
    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
        folders_and_frames: Dict[str, list],
        customer_name: str,
        image_upload_date: str,
    ) -> None:
        """
        This class extends YOLOInference class to run inference on images using a pre-trained YOLO model.
        The extension consist on a  de-fisheye. For more details check the docstring of parent class.

        Parameters
        ----------
        images_folder: str
            Location of images to run inference on. If the location contains
            sub-folders, this folder structure will be preserved in the output.
        output_folder: str
            Location where output (annotation labels and possible images) will
            be stored.
        model_path: str
            Location of the pre-trained YOLOv8 model.
        inference_settings: Dict
            Settings for the model, which contains:
                model_params: Dict
                    Inference parameters for the YOLOv8 model:
                        img_size, save_img_flag, save_txt_flag, save_conf_flag, conf
                target_classes: List
                    List of target classes for which bounding boxes will be predicted.
                sensitive_classes: List
                    List of sensitive classes which will be blurred in output images.
                target_classes_conf: Optional[float] = None
                    Optional: confidence threshold for target classes. Only detections
                    above this threshold will be considered. If omitted,
                    inference_param["conf"] will be used.
                sensitive_classes_conf: Optional[float] = None
                    Optional: confidence threshold for sensitive classes. Only
                    detections above this threshold will be considered. If omitted,
                    inference_param["conf"] will be used.
                output_image_size: Optional[Tuple[int, int]] = None
                    Optional: output images will be resized to these (width, height)
                    dimensions if set.
                defisheye_flag: bool = False
                    Whether or not to apply distortion correction to the input images.
                defisheye_params: Dict = {}
                    If defisheye_flag is True, these distortion correction parameters
                    will be used. Contains "camera_matrix", "distortion_params", and
                    "input_image_size" (size of images used to compute these
                    parameters).
                save_images: bool = False
                    Whether or not to save the output images.
                save_labels: bool = True
                    Whether or not to save the annotation labels.
                save_all_images: bool = False
                    Whether to save all processed images (True) or only those containing
                    objects belonging to one of the target classes (False).
                save_images_subfolder: Optional[str] = None
                    Optional: sub-folder in which to store output images.
                save_labels_subfolder: Optional[str] = None
                    Optional: sub-folder in which to store annotation labels.
                batch_size: int = 1
                    Batch size for inference.
                defisheye_flag: bool = False
                    Whether or not to apply distortion correction to the input images.
                defisheye_params: Dict = {}
                    If defisheye_flag is True, these distortion correction parameters
                    will be used. Contains "camera_matrix", "distortion_params", and
                    "input_image_size" (size of images used to compute these
                    parameters).
        folders_and_frames: Dict[str, list]
            Dictionary containing the folder structure and frames for each folder.
        customer_name: str
            Customer name for the images.
        image_upload_date: str
            Date when the images were uploaded.
        """
        super().__init__(
            images_folder=images_folder,
            output_folder=output_folder,
            model_path=model_path,
            inference_settings=inference_settings,
        )
        self.folders_and_frames = folders_and_frames
        self.customer_name = customer_name
        self.image_upload_date = image_upload_date

    def _process_detections(
        self, model_results: List[Results], image_paths: List[str]
    ) -> None:
        """
        Process the BaaS inference Results objects extending the YOLOInference class.
        In addition it updates the database with the results.

        Parameters
        ----------
        model_results: List[Results]
            List of YOLOv8 inference Results objects.
        image_paths: List[str]
            List of input image paths corresponding to the Results.
        """
        super()._process_detections(model_results, image_paths)
        batch_detection_info = []
        for result, image_path in zip(model_results, image_paths):
            p = Path(image_path)
            image_filename = (
                Path("/".join(p.parts[p.parts.index("wd") + 2 :]))
                if "wd" in p.parts
                else p
            )

            result_detections = result.boxes
            for idx, cls in enumerate(result_detections.cls):
                batch_detection_info.append(
                    {
                        "image_customer_name": self.customer_name,
                        "image_upload_date": self.image_upload_date,
                        "image_filename": str(image_filename),
                        "has_detection": True,
                        "class_id": int(cls.item()),
                        "x_norm": float(result_detections.xyxy[idx][0].item()),
                        "y_norm": float(result_detections.xyxy[idx][1].item()),
                        "w_norm": float(result_detections.xyxy[idx][2].item()),
                        "h_norm": float(result_detections.xyxy[idx][3].item()),
                        "image_width": int(result.orig_shape[1]),
                        "image_height": int(result.orig_shape[0]),
                        "run_id": "",
                        "conf_score": float(result_detections.conf[idx].item()),
                    }
                )

            db_connector = create_db_connector()
            db_connector.create_connection()
            try:
                with db_connector.managed_session() as session:
                    if batch_detection_info:
                        session.bulk_insert_mappings(
                            DetectionInformation, batch_detection_info
                        )
                    else:
                        empty_detection = DetectionInformation(
                            image_customer_name=self.customer_name,
                            image_upload_date=self.image_upload_date,
                            image_filename=str(image_filename),
                            has_detection=False,
                            class_id=None,
                            x_norm=None,
                            y_norm=None,
                            w_norm=None,
                            h_norm=None,
                            image_width=None,
                            image_height=None,
                            run_id="",
                            conf_score=None,
                        )
                        session.add(empty_detection)
                    image_processing_status = ImageProcessingStatus(
                        image_filename=str(image_filename),
                        image_upload_date=self.image_upload_date,
                        image_customer_name=self.customer_name,
                        processing_status="processed",
                    )
                    session.merge(image_processing_status)
            except SQLAlchemyError as e:
                db_connector.close_connection()
                raise e
            db_connector.close_connection()
