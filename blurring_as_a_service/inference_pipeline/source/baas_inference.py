import logging
from typing import Dict

from yolo_model_development_kit.inference_pipeline.source.YOLO_inference import (
    YOLOInference,
)

logger = logging.getLogger("inference_pipeline")


class BaaSInference(YOLOInference):
    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
        folders_and_frames: Dict[str, list],
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
                    Whether to save all processed images (TRue) or only those containing
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
        """
        self.folders_and_frames = folders_and_frames
        super().__init__(
            images_folder=images_folder,
            output_folder=output_folder,
            model_path=model_path,
            inference_settings=inference_settings,
        )
