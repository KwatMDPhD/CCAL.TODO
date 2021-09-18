from pandas import DataFrame, Index

from ..plot import plot_heat_map, plot_plotly
from .get_1d_grid import get_1d_grid


def plot(co_po_di, ve, na_=(), pa=""):

    co__ = get_1d_grid(co_po_di)

    nu_po_di = ve.reshape([co_.size for co_ in co__])

    n_di = co_po_di.shape[1]

    if len(na_) == n_di + 1:

        nav = na_.pop()

    else:

        na_ = ["Dimension {}".format(ie) for ie in range(n_di)]

        nav = "Dimension {}".format(n_di + 1)

    for na, co_ in zip(na_, co__):

        print(
            "{} (grid): size={} min={:.2e} max={:.2e}".format(
                na, co_.size, co_.min(), co_.max()
            )
        )

    print("{}: min={:.2e} max={:.2e}".format(nav, ve.min(), ve.max()))

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
                            "text": nav,
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
                    "text": nav,
                },
            },
            pa=pa,
        )
