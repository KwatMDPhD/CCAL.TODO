from KDEpy import FFTKDE, NaiveKDE, TreeKDE

from ..constant import float_resolution
from ..grid import make_1d_grid, make_nd_grid, plot


def get_density(nu_po_di, me="fft", co__=(), pl=True, di_=(), **ke_va):

    n_po, n_di = nu_po_di.shape

    if len(co__) != n_di:

        print("Making coordinates...")

        co__ = [make_1d_grid(nu_.min(), nu_.max(), 1 / 3, 8) for nu_ in nu_po_di.T]

    co_po_di = make_nd_grid(co__)

    de_ = (
        {"naive": NaiveKDE, "tree": TreeKDE, "fft": FFTKDE,}[
            me
        ](**ke_va)
        .fit(nu_po_di)
        .evaluate(grid_points=co_po_di)
        .clip(min=float_resolution)
    )

    if pl:

        plot(co_po_di, de_, nu="Density", di_=di_)

    return co_po_di, de_
