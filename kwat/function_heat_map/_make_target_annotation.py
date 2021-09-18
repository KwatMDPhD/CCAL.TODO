from ..string import trim
from .ANNOTATION import ANNOTATION


def _make_target_annotation(y, ro):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(trim(ro)),
            **ANNOTATION,
        }
    ]
