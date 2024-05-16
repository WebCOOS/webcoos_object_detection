
from prometheus_client import (
    make_asgi_app,
    CollectorRegistry,
    multiprocess,
    Counter
)
from model_version import (
    ModelFramework,
    TFModelName,
    TFModelVersion,
    YOLOModelName,
    YOLOModelVersion
)


OBJECT_CLASSIFICATION_COUNTER = Counter(
    'object_classification_counter',
    'Count of classifications',
    [
        'model_framework',
        'model_name',
        'model_version',
        'classification_name',
    ]
)

# Per: <https://prometheus.github.io/client_python/instrumenting/labels/>
#   Metrics with labels are not initialized when declared, because the client
#   can’t know what values the label can have. It is recommended to initialize
#   the label values by calling the .labels() method alone:
#
#       c.labels('get', '/')

LABELS = (
    ( ModelFramework.TF, TFModelName.yolo_detector, TFModelVersion.two, 'seal' ),
    ( ModelFramework.TF, TFModelName.yolo_detector, TFModelVersion.three, 'seal' ),
    ( ModelFramework.YOLO, YOLOModelName.best_yolo, YOLOModelVersion.one, 'seal' ),
)

for ( fw, mdl, ver, cls_name ) in LABELS:
    OBJECT_CLASSIFICATION_COUNTER.labels(
        fw.name,
        mdl.value,
        ver.value,
        cls_name,
    )


def make_metrics_app():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector( registry )
    return make_asgi_app( registry = registry )


def increment_yolo_counter(
    fw: str,
    mdl_name: str,
    mdl_version: str,
    cls_name: str
):
    OBJECT_CLASSIFICATION_COUNTER.labels(
        fw,
        mdl_name,
        mdl_version,
        cls_name
    ).inc()
