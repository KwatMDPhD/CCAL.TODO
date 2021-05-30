from numpy import asarray, diff, linspace, meshgrid, unique
from pandas import DataFrame, Index

from .plot import plot_heat_map, plot_plotly


def make_1d_grid(lo, hi, fr, n_co):

    ex = (hi - lo) * fr

    lo -= fr

    hi += fr

    return linspace(lo, hi, n_co)


def reflect_1d_grid(co_, re):

    co2_ = co_.copy()

    for ie, co in enumerate(co_):

        if co < re:

            co2_[ie] += (re - co) * 2

        else:

            co2_[ie] -= (co - re) * 2

    return co2_


def get_1d_grid_resolution(co_):

    return diff(unique(co_)).min()


def get_1d_grid(co_po_di):

    return [unique(co_) for co_ in co_po_di.T]


def make_nd_grid(co__):

    return asarray([co_po_di.ravel() for co_po_di in meshgrid(*co__, indexing="ij")]).T


def plot(co_po_di, nu_, di_=(), nu="Number", pa=""):

    n_di = co_po_di.shape[1]

    if len(di_) != n_di:

        di_ = ["Dimension {}".format(ie) for ie in range(n_di)]

    co__ = get_1d_grid(co_po_di)

    nu_po_di = nu_.reshape([co_.size for co_ in co__])

    for ie, co_ in enumerate(co__):

        print(
            "Dimension {} grid: size={} min={:.2e} max={:.2e}".format(
                ie + 1, co_.size, co_.min(), co_.max()
            )
        )

    print("Number: min={:.2e} max={:.2e}".format(nu_po_di.min(), nu_po_di.max()))

    if n_di == 1:

        plot_plotly(
            {
                "data": [
                    {
                        "y": nu_po_di,
                        "x": co__[0],
                    }
                ],
                "layout": {
                    "yaxis": {
                        "title": {
                            "text": nu,
                        },
                    },
                    "xaxis": {
                        "title": {
                            "text": di_[0],
                        },
                    },
                },
            },
            pa=pa,
        )

    elif n_di == 2:

        plot_heat_map(
            DataFrame(
                nu_po_di,
                Index(("{:.2e} *".format(co) for co in co__[0]), name=di_[0]),
                Index(("* {:.2e}".format(co) for co in co__[1]), name=di_[1]),
            ),
            layout={
                "title": {
                    "text": nu,
                },
            },
            pa=pa,
        )
