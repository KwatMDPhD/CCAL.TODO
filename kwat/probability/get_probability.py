from numpy import product

from ..density import get_density
from ..grid import get_1d_grid_resolution, plot


def get_probability(nu_po_di, co__=(), pl=True, na_=(), **ke_ar):

    co_po_di, de_ = get_density(nu_po_di, co__=co__, pl=pl, na_=na_, **ke_ar)

    pr_ = de_ / (
        de_.sum() * product([get_1d_grid_resolution(co_) for co_ in co_po_di.T])
    )

    if pl:

        plot(co_po_di, pr_, na_=list(na_) + ["Probability"])

    return co_po_di, pr_
