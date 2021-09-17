from numpy import apply_along_axis
from pandas import DataFrame, Index

from ..array import normalize
from ..cluster import cluster
from ..constant import GOLDEN_RATIO
from ..plot import plot_heat_map, plot_plotly


def make_factor_label(re):

    na = "Factor"

    return Index(data=("{} {} {}".format(na, re, ie + 1) for ie in range(re)), name=na)


def plot(
    maw_,
    mah_,
    er_ie_it=None,
    si=640,
    pa="",
):

    sig = si * GOLDEN_RATIO

    faxis = {
        "dtick": 1,
    }

    for ie, maw in enumerate(maw_):

        maw = apply_along_axis(normalize, 1, maw[cluster(maw)[0], :], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}w_{}.html".format(pa, ie)

        plot_heat_map(
            DataFrame(
                data=maw,
                # index=Index(data=ro__[ie], name=ron[ie]),
                columns=make_factor_label(maw.shape[1]),
            ),
            layout={
                "height": sig,
                "width": si,
                "title": {
                    "text": "W {}".format(ie),
                },
                "xaxis": faxis,
            },
            pa=pa2,
        )

    for ie, mah in enumerate(mah_):

        mah = apply_along_axis(normalize, 0, mah[:, cluster(mah.T)[0]], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}h_{}.html".format(pa, ie)

        plot_heat_map(
            DataFrame(
                data=mah,
                index=make_factor_label(mah.shape[0]),
                # columns=Index(data=co__[ie], name=con[ie]),
            ),
            layout={
                "height": si,
                "width": sig,
                "title": {
                    "text": "H {}".format(ie),
                },
                "yaxis": faxis,
            },
            pa=pa2,
        )

    if er_ie_it is not None:

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
                    for ie, er_ in enumerate(er_ie_it)
                ],
                "layout": {
                    "xaxis": {
                        "title": "Iteration",
                    },
                    "yaxis": {
                        "title": "Error",
                    },
                    "annotations": [
                        {
                            "x": er_.size - 1,
                            "y": er_[-1],
                            "text": "{:.2e}".format(er_[-1]),
                        }
                        for er_ in er_ie_it
                    ],
                },
            },
            pa=pa2,
        )
