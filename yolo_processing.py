import os
from typing import List, Set, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from score import ClassificationModelResult, BoundingBoxPoint
from model_version import (
    ModelFramework,
    YOLOModelName,
    YOLOModelVersion,
    YOLOModelObjectClassification
)
from metrics import increment_detection_counter, increment_object_counter
import logging

logger = logging.getLogger( __name__ )
width = 896
height = 896
DEFAULT_YOLO_THRESHOLD = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX

MODEL_FOLDER = Path(os.environ.get(
    "MODEL_DIRECTORY",
    str(Path(__file__).parent)
))

device = torch.device('cpu')
if torch.cuda.is_available():
    logger.warning(
        "GPU/CUDA resources available"
    )
    device = torch.device('cuda')
else:
    logger.warning(
        "GPU/CUDA resources not available (according to torch.cuda.is_available)"
    )

YOLO_MODELS = {
    # public-facing model name
    "yolo": {
        # public-facing model version
        "v8n": YOLO(
            str(
                MODEL_FOLDER \
                # tech stack (yolo, torchvision, tensorflow, etc.)
                / "ultralytics" \
                # model name (within the tech stack)
                / "yolo" \
                # model version (for that model)
                / "v8n" \
                # versioned file for this model
                / "yolov8n.pt"
            ),
        ),
    }
}

# Pre-emptively move models to selected device
for ( _, v ) in YOLO_MODELS["yolo"].items():
    v.to( device )


#SEAL_CLASSIFICATION = 0.0


def yolo_process_image(
    yolo_model: YOLO,
    output_path: Path,
    model: Union[YOLOModelName, str],
    version: Union[YOLOModelVersion, str],
    name: str,
    bytedata: bytes,
    confidence_threshold: float = None,
    cls_names_valid: List[YOLOModelObjectClassification] = None,
    group: str = None,
    asset: str = None,
):

    assert yolo_model, \
        f"Must have yolo_model passed to {yolo_process_image.__name__}"

    assert output_path and isinstance( output_path, Path ), \
        f"output_path parameter for {yolo_process_image.__name__} is not Path"

    assert output_path.exists() and output_path.is_dir(), \
        (
            f"output_path parameter for {yolo_process_image.__name__} must exist "
            "and be a directory"
        )

    assert isinstance( model, ( YOLOModelName, str ) )
    assert isinstance( version, ( YOLOModelVersion, str ) )

    if( isinstance( model, YOLOModelName ) ):
        model = model.value

    if( isinstance( version, YOLOModelVersion ) ):
        version = version.value

    if confidence_threshold is None:
        confidence_threshold = DEFAULT_YOLO_THRESHOLD

    assert confidence_threshold >= 0.0 and confidence_threshold <= 1.0

    if (
        cls_names_valid is None
        or len( [ x for x in cls_names_valid if x ]) <= 0
    ):
        # Default to the full list of available class names
        cls_names_valid = list( YOLOModelObjectClassification )

    assert all(
        [
            isinstance( x, YOLOModelObjectClassification )
            for x in cls_names_valid
        ]
    )

    ret: ClassificationModelResult = ClassificationModelResult(
        ModelFramework.ultralytics.name,
        model,
        version
    )

    output_file = output_path / model / str(version) / name

    npdata = np.asarray(bytearray(bytedata), dtype="uint8")
    frame = cv2.imdecode(npdata, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_boxes = frame

    #use YOLOv8
    results = yolo_model.predict(frame, conf = 0.1)

    # If any score is above threshold, flag it as detected
    detected = False

    yolo_object_class: YOLOModelObjectClassification = None

    object_classes_detected: Set[YOLOModelObjectClassification] = set()

    for result in results:
        #for score, cls, cls_name, bbox in zip(result.boxes.conf, result.boxes.cls, result.names, result.boxes.xyxy):
        for box in result.boxes:

            score = box.conf.item()
            cls = int(box.cls.item())
            cls_name = yolo_model.names[cls]

            if score < confidence_threshold:
                continue

            try:
                yolo_object_class = YOLOModelObjectClassification( cls_name )
            except ValueError:
                logger.warning(
                    f"Detected class of '{cls_name}', but not among accepted "
                    "classes, ignoring."
                )
                continue

            if yolo_object_class not in cls_names_valid:
                # Skip object, even if it meets with our threshold, as we aren't
                # interested for this particular detection
                continue

            detected = True

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            h, w, _ = frame.shape

            y_min = int(max(1, y1))
            x_min = int(max(1, x1))
            y_max = int(min(h, y2))
            x_max = int(min(w, x2))

            if yolo_object_class is not None:

                label = cls_name + ": " + ": {:.2f}%".format(score * 100)
                img_boxes = cv2.rectangle(img_boxes, (x_min, y_max), (x_max, y_min), (0, 0, 255), 2)
                cv2.putText(img_boxes, label, (x_min, y_max - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ret.add(
                    classification_name=yolo_object_class.value,
                    classification_score=score,
                    bbox=(
                        BoundingBoxPoint( x_min, y_min ),
                        BoundingBoxPoint( x_max, y_max ),
                    )
                )

                # Update object metrics
                increment_object_counter(
                    ModelFramework.ultralytics.name,
                    model,
                    version,
                    yolo_object_class.value,
                    group,
                    asset,
                )

                # Track which of the object classes were detected for later
                # metrics counters
                object_classes_detected.add(
                    yolo_object_class
                )

    # outp = cv2.resize(img_boxes, (1280, 720))

    if detected is True:

        if( object_classes_detected ):

            for ocd in object_classes_detected:

                # Update Prometheus metrics for each of the classes that were
                # detected
                increment_detection_counter(
                    ModelFramework.ultralytics.name,
                    model,
                    version,
                    ocd.value,
                    group,
                    asset,
                )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), img_boxes )
        return ( str(output_file), ret )

    return ( None, ret  )
