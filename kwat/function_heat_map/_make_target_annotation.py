from ..string import trim
from .ANNOTATION_TEMPLATE import ANNOTATION_TEMPLATE


def _make_target_annotation(y, ro):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(trim(ro)),
            **ANNOTATION_TEMPLATE,
        }
    ]
