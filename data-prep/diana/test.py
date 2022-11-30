from utils import CocoToYoloConverter

source = "in:coco-format/blur_v0.1/blur-annotations.json"
output_dir = "out:yolo-format/blur_v0.1"

converter = CocoToYoloConverter(source, output_dir)
converter.convert()