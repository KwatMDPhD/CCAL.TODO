from numpy import apply_along_axis
from pandas import DataFrame

from ..array import normalize
from ..cluster import cluster
from ..constant import golden_ratio
from ..plot import plot_heat_map, plot_plotly
from .label_factor import label_factor


def plot(
    wm_,
    hm_,
    er_ma_it=None,
    si=640,
    pa="",
):

    sig = si * golden_ratio

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
                columns=label_factor(wm.shape[1]),
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

    for ie, hm in enumerate(hm_):

        hm = apply_along_axis(normalize, 0, hm[:, cluster(hm.T)[0]], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}h_{}.html".format(pa, ie)

        plot_heat_map(
            DataFrame(
                data=hm,
                index=label_factor(hm.shape[0]),
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
                        for er_ in er_ma_it
                    ],
                },
            },
            pa=pa2,
        )
