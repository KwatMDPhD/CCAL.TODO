from ..string import trim
from .annotation import annotation


def _make_target_annotation(y, text):

    return [
        {
            "y": y,
            "x": 0,
            "xanchor": "right",
            "text": "<b>{}</b>".format(trim(text)),
            **annotation,
        }
    ]
