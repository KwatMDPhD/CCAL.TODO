from ..string import trim
from ._get_statistic_x import _get_statistic_x
from .ANNOTATION import ANNOTATION


def _make_data_annotations(y, la, he, text_, fu):

    annotations = []

    n_ch = 27

    if la:

        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION,
                }
            )

    y -= he

    for ie1 in range(text_.size):

        annotations.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": "{}".format(trim(text_[ie1])),
                **ANNOTATION,
            }
        )

        sc, ma, pv, qv = fu[ie1]

        for ie2, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):

            annotations.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie2),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION,
                }
            )

        y -= he

    return annotations
