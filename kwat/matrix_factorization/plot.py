from numpy import apply_along_axis

from ..array import normalize
from ..cluster import cluster
from ..constant import golden_ratio
from ..plot import plot_heat_map, plot_plotly
from .make_factor_label import make_factor_label


def plot(
    wm_,
    hm_,
    ro__,
    co__,
    ron,
    con,
    er__,
    si=640,
    pa="",
):

    fsi = si * golden_ratio

    faxis = {
        "dtick": 1,
    }

    for iew, wm in enumerate(wm_):

        wm = apply_along_axis(normalize, 1, wm[cluster(wm)[0], :], "-0-")

        if pa == "":

            pa2 = pa

        else:

            pa2 = "{}w_{}.html".format(pa, iew)

        plot_heat_map(
            wm,
            ro__[iew],
            make_factor_label(wm.shape[1]),
            ron[iew],
            "Factor",
            layout={
                "height": fsi,
                "width": si,
                "title": {
                    "text": "W {}".format(iew),
                },
                "xaxis": faxis,
            },
            pa=pa2,
        )

    for ieh, hm in enumerate(hm_):

        hm = apply_along_axis(normalize, 0, hm[:, cluster(hm.T)[0]], "-0-")

        if pa == "":

            pa2 = None

        else:

            pa2 = "{}h_{}.html".format(pa, iew)

        plot_heat_map(
            hm,
            make_factor_label(hm.shape[0]),
            co__[ieh],
            "Factor",
            con[ieh],
            layout={
                "height": si,
                "width": fsi,
                "title": {
                    "text": "H {}".format(ieh),
                },
                "yaxis": faxis,
            },
            pa=pa2,
        )

    if pa == "":

        pa2 = None

    else:

        pa2 = "{}error.html".format(pa)

    plot_plotly(
        {
            "data": [
                {
                    "name": index,
                    "y": er_,
                }
                for index, er_ in enumerate(er__)
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
                    for er_ in er__
                ],
            },
        },
        pa=pa2,
    )
