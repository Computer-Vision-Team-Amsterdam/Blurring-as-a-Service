from enum import IntFlag, auto, unique
from functools import reduce


@unique
class PipelineFlag(IntFlag):
    NONE = 0
    CONVERT_AZURE_COCO_TO_YOLO = auto()
    CONVERT_AZURE_COCO_TO_COCO = auto()
    CREATE_METADATA = auto()
    VALIDATE_MODEL = auto()
    COCO_EVALUATION = auto()
    STORE_MODEL = auto()

    @classmethod
    def all(cls):
        return reduce(lambda m1, m2: m1 | m2, [m for m in cls.__members__.values()])
