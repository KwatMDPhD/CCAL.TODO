from KDEpy import TreeKDE

from ..constant import FLOAT_RESOLUTION
from ..grid import make_1d_grid, make_nd_grid, plot
from .get_bandwidth import get_bandwidth


def get_density(nu_po_di, ba_=(), co__=(), pl=True, di_=()):

    n_po, n_di = nu_po_di.shape

    if len(ba_) != n_di:

        ba_ = get_bandwidth(nu_po_di)

    if len(co__) != n_di:

        co__ = [make_1d_grid(nu_.min(), nu_.max(), 1 / 3, 8) for nu_ in nu_po_di.T]

    co_po_di = make_nd_grid(co__)

    de_ = (
        TreeKDE(bw=ba_ * n_po)
        .fit(nu_po_di)
        .evaluate(grid_points=co_po_di)
        .clip(min=FLOAT_RESOLUTION)
    )

    if pl:

        plot(co_po_di, de_, nu="Density", di_=di_)

    return co_po_di, de_
