from numpy import apply_along_axis
from pandas import DataFrame, Index

from ..array import normalize
from ..cluster import cluster
from ..constant import GOLDEN_RATIO
from ..plot import plot_heat_map, plot_plotly


def make_factor_label(re):

    na = "Factor"

    return Index(data=("{} {}_{}".format(na, re, ie) for ie in range(re)), name=na)


def plot(
    wm_,
    hm_,
    er_ma_it=None,
    si=640,
    pa="",
):

    sig = si * GOLDEN_RATIO

    faxis = {
        "dtick": 1,
    }

    for ie, wm in enumerate(wm_):

        wm = apply_along_axis(normalize, 1, wm[cluster(wm)[0], :], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}w_{}.html".format(pa, ie)

        plot_heat_map(
            DataFrame(
                data=wm,
                # index=Index(data=ro__[ie], name=ron[ie]),
                COLUMNS=make_factor_label(wm.shape[1]),
            ),
            LAYOUT_TEMPLATE={
                "height": sig,
                "width": si,
                "title": {
                    "text": "W {}".format(ie),
                },
                "xaxis": faxis,
            },
            pa=pa2,
        )

    for ie, hm in enumerate(hm_):

        hm = apply_along_axis(normalize, 0, hm[:, cluster(hm.T)[0]], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}h_{}.html".format(pa, ie)

        plot_heat_map(
            DataFrame(
                data=hm,
                index=make_factor_label(hm.shape[0]),
                # COLUMNS=Index(data=co__[ie], name=con[ie]),
            ),
            LAYOUT_TEMPLATE={
                "height": si,
                "width": sig,
                "title": {
                    "text": "H {}".format(ie),
                },
                "yaxis": faxis,
            },
            pa=pa2,
        )

    if er_ma_it is not None:

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}error.html".format(pa)

        plot_plotly(
            {
                "data": [
                    {
                        "name": ie,
                        "y": er_,
                    }
                    for ie, er_ in enumerate(er_ma_it)
                ],
                "LAYOUT_TEMPLATE": {
                    "xaxis": {
                        "title": "Iteration",
                    },
                    "yaxis": {
                        "title": "Error",
                    },
                    "ANNOTATION_TEMPLATEs": [
                        {
                            "x": er_.size - 1,
                            "y": er_[-1],
                            "text": "{:.2e}".format(er_[-1]),
                        }
                        for er_ in er_ma_it
                    ],
                },
            },
            pa=pa2,
        )
