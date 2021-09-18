from ..string import trim
from .ANNOTATION import ANNOTATION


def _get_x(ie):

    return 1.08 + ie / 6.4


def _make_data_annotation(y, la, he, ro_, st):

    annotations = []

    if la:

        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):

            annotations.append(
                {
                    "y": y,
                    "x": _get_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION,
                }
            )

    y -= he

    for iey in range(ro_.size):

        annotations.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": "{}".format(trim(ro_[iey])),
                **ANNOTATION,
            }
        )

        sc, ma, pv, qv = st[iey]

        for iex, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):

            annotations.append(
                {
                    "y": y,
                    "x": _get_x(iex),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION,
                }
            )

        y -= he

    return annotations
