import copy
import json


class AzureCocoToCocoConverter:
    def __init__(
        self, azureml_file: str, output_file: str, new_width: float, new_height: float
    ):
        self._filename = azureml_file
        self._output_file = output_file
        self.new_width = new_width
        self.new_height = new_height

        with open(azureml_file) as f:
            self._input = json.load(f)

    def _add_key(self, key: str, value) -> None:
        """
        Adds key to the input file

        Parameters
        ----------
        key name of key to be added in _input[annotations]
        value corresponding value for the key

        Returns
        -------

        """

        for i, _ in enumerate(self._input["annotations"]):
            self._input["annotations"][i][key] = value

    def _calculate_area(self) -> None:
        anns_copy = copy.deepcopy(self._input["annotations"])

        for i, ann in enumerate(anns_copy):
            x1, y1, x2, y2 = ann["bbox"]
            self._input["annotations"][i]["area"] = (x2 - x1) * (y2 - y1)

    def _to_absolute(self) -> None:
        """
        Converts normalized bbox values to absolute values.
        """

        for i, ann in enumerate(self._input["annotations"]):
            bbox_absolute_values = []
            for x, y in zip(ann["bbox"][::2], ann["bbox"][1::2]):
                bbox_absolute_values.append(x * self.new_width)
                bbox_absolute_values.append(y * self.new_height)
            self._input["annotations"][i]["bbox"] = bbox_absolute_values

    def _save(self):
        with open(self._output_file, "w") as f:
            json.dump(self._input, f)

    def _update_categories(self):
        if len(self._input["categories"]) == 2:
            for i, _ in enumerate(self._input["categories"]):
                self._input["categories"][i]["id"] = (
                    self._input["categories"][i]["id"] - 1
                )
            for i, _ in enumerate(self._input["annotations"]):
                self._input["annotations"][i]["category_id"] = self._input[
                    "annotations"
                ][i]["category_id"]

    def _update_ids_and_image_ids(self):
        new_images = []
        new_annotations = []
        for i, image in enumerate(self._input["images"]):
            # collect corresponding annotations
            annotations = [
                annotation
                for annotation in self._input["annotations"]
                if annotation["image_id"] == image["id"]
            ]

            # update id of image
            new_filename = image["file_name"].split("/")[-1].split(".")[0]
            image["id"] = new_filename
            new_images.append(image)

            # update image_id of annotations
            for annotation in annotations:
                annotation["image_id"] = image["id"]
                annotation["category_id"] = annotation["category_id"] - 1
                new_annotations.append(annotation)

        self._input["images"] = new_images
        self._input["annotations"] = new_annotations

    def convert(self) -> None:
        self._add_key(key="iscrowd", value=0)
        self._add_key(key="segmentation", value=[])
        self._to_absolute()  # we must calculate area based on absolute values
        self._calculate_area()
        self._update_categories()
        self._update_ids_and_image_ids()
        self._save()
