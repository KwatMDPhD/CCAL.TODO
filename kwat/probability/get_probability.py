from numpy import product

from ..density import get_density
from ..grid import get_1d_grid_resolution, plot as grid_plot


def get_probability(nu_po_di, co__=(), pl=True, di_=(), **ff):

    co_po_di, de_ = get_density(nu_po_di, co__=co__, pl=pl, di_=di_, **ff)

    pr_ = de_ / (
        de_.sum() * product([get_1d_grid_resolution(co_) for co_ in co_po_di.T])
    )

    if pl:

        grid_plot(co_po_di, pr_, di_=di_, nu="Probability")

    return co_po_di, pr_
