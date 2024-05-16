from enum import Enum


class ModelFramework(str, Enum):
    TF = "TF"
    YOLO = "YOLO"


class TFModelName(str, Enum):
    yolo_detector = "yolo_detector"


class TFModelVersion(str, Enum):
    two = "2"
    three = "3"


class YOLOModelName(str, Enum):
    best_yolo = "best_yolo"


class YOLOModelVersion(str, Enum):
    one = "1"
