from ..string import trim
from .ANNOTATION_TEMPLATE import ANNOTATION_TEMPLATE


def _get_statistic_x(ie):

    return 1.08 + ie / 6.4


def _make_data_annotation(yc, la, he, ro_, st):

    an_ = []

    if la:

        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):

            an_.append(
                {
                    "y": yc,
                    "x": _get_statistic_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION_TEMPLATE,
                }
            )

    yc -= he

    for iey in range(ro_.size):

        an_.append(
            {
                "y": yc,
                "x": 0,
                "xanchor": "right",
                "text": "{}".format(trim(ro_[iey])),
                **ANNOTATION_TEMPLATE,
            }
        )

        sc, ma, pv, qv = st[iey]

        for iex, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):

            an_.append(
                {
                    "y": yc,
                    "x": _get_statistic_x(iex),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION_TEMPLATE,
                }
            )

        yc -= he

    return an_
