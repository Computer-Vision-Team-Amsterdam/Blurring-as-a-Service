import json


def _update_category(annotation):
    annotation["category_id"] = annotation["category_id"] - 1
    return annotation


class CocoToYoloConverter:
    def __init__(self, coco_file: str, yolo_folder: str):
        """
        :param coco_file: path to input coco file
        :param yolo_folder: path to folder where to store all .txt files
        """
        self._output_dir = yolo_folder

        with open(coco_file) as f:
            self._input = json.load(f)
        f.close()

    def _write_to_txt(self, image_name, per_image_annotations):
        # line = [category, x, ,y, width, height]
        lines = [f'{annotation["category_id"]} {" ".join(map(str, annotation["bbox"]))}'
                 for annotation in per_image_annotations]
        with open(f'{self._output_dir}/{image_name}.txt', 'w') as f:
            f.write('\n'.join(lines))

    def convert(self):
        # collect images
        for i, image in enumerate(self._input["images"]):
            # collect corresponding annotations
            annotations = [annotation for annotation in self._input["annotations"]
                           if annotation["image_id"] == image["id"]]
            # update category
            annotations = [_update_category(annotation) for annotation in annotations]

            # write annotations to txt file. image_name is Location/images/TMXblabla.jpg
            self._write_to_txt(image_name=image["file_name"].split("/")[-3],
                               per_image_annotations=annotations)