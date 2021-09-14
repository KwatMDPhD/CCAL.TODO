from ..string import trim
from .ANNOTATION_TEMPLATE import ANNOTATION_TEMPLATE


def _get_statistic_x(ie):

    return 1.08 + ie / 6.4


def _make_data_annotation(y, la, he, text_, fu):

    ANNOTATION_TEMPLATEs = []

    n_ch = 27

    if la:

        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):

            ANNOTATION_TEMPLATEs.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie),
                    "xanchor": "center",
                    "text": "<b>{}</b>".format(text),
                    **ANNOTATION_TEMPLATE,
                }
            )

    y -= he

    for ie1 in range(text_.size):

        ANNOTATION_TEMPLATEs.append(
            {
                "y": y,
                "x": 0,
                "xanchor": "right",
                "text": "{}".format(trim(text_[ie1])),
                **ANNOTATION_TEMPLATE,
            }
        )

        sc, ma, pv, qv = fu[ie1]

        for ie2, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):

            ANNOTATION_TEMPLATEs.append(
                {
                    "y": y,
                    "x": _get_statistic_x(ie2),
                    "xanchor": "center",
                    "text": text,
                    **ANNOTATION_TEMPLATE,
                }
            )

        y -= he

    return ANNOTATION_TEMPLATEs
