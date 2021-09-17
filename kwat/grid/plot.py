from pandas import DataFrame, Index

from ..plot import plot_heat_map, plot_plotly
from .get_1d_grid import get_1d_grid


def plot(co_po_di, ve, nu="Number", na_=(), pa=""):

    n_di = co_po_di.shape[1]

    if len(na_) != n_di:

        na_ = ["Dimension {}".format(ie) for ie in range(n_di)]

    co__ = get_1d_grid(co_po_di)

    for ie, co_ in enumerate(co__):

        print(
            "Dimension {} grid: size={} min={:.2e} max={:.2e}".format(
                ie + 1, co_.size, co_.min(), co_.max()
            )
        )

    print("Number: min={:.2e} max={:.2e}".format(ve.min(), ve.max()))

    nu_po_di = ve.reshape([co_.size for co_ in co__])

    if n_di == 1:

        plot_plotly(
            {
                "data": [
                    {
                        "x": co__[0],
                        "y": nu_po_di,
                    }
                ],
                "layout": {
                    "xaxis": {
                        "title": {
                            "text": na_[0],
                        },
                    },
                    "yaxis": {
                        "title": {
                            "text": nu,
                        },
                    },
                },
            },
            pa=pa,
        )

    elif n_di == 2:

        plot_heat_map(
            DataFrame(
                data=nu_po_di,
                index=Index(
                    data=["{:.2e} *".format(co) for co in co__[0]], name=na_[0]
                ),
                columns=Index(
                    data=["* {:.2e}".format(co) for co in co__[1]], name=na_[1]
                ),
            ),
            layout={
                "title": {
                    "text": nu,
                },
            },
            pa=pa,
        )
