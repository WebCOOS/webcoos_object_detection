from enum import Enum


class ModelFramework(str, Enum):
    ULTRALYTICS = "ULTRALYTICS"

class YOLOModelName(str, Enum):
    yolo = "yolo"


class YOLOModelVersion(str, Enum):
    v8n = "v8n"
