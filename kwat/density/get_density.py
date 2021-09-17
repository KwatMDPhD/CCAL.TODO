from KDEpy import FFTKDE

from ..constant import FLOAT_RESOLUTION
from ..grid import make_1d_grid, make_nd_grid, plot


def get_density(nu_po_di, co__=(), pl=True, na_=(), **di):

    n_po, n_di = nu_po_di.shape

    if len(co__) != n_di:

        print("Making coordinates")

        co__ = [make_1d_grid(ve.min(), ve.max(), 1 / 3, 8) for ve in nu_po_di.T]

    co_po_di = make_nd_grid(co__)

    de_ = (
        FFTKDE(**di)
        .fit(nu_po_di)
        .evaluate(grid_points=co_po_di)
        .clip(min=FLOAT_RESOLUTION)
    )

    if pl:

        plot(co_po_di, de_, nu="Density", na_=na_)

    return co_po_di, de_
