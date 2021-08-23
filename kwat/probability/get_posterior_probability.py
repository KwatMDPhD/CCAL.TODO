from numpy import absolute, apply_along_axis, isnan, nan, s_, unique

from ..grid import get_1d_grid, get_1d_grid_resolution, plot as grid_plot
from ._get_probability import _get_probability
from .get_probability import get_probability


def get_posterior_probability(nu_po_di, ta=nan, co__=(), pl=True, di_=(), **ff):

    co_po_di, pr_ = get_probability(nu_po_di, co__=co__, pl=pl, di_=di_, **ff)

    ta_ = co_po_di[:, -1]

    pr___ = pr_.reshape([co_.size for co_ in get_1d_grid(co_po_di)])

    po___ = apply_along_axis(_get_probability, -1, pr___) * get_1d_grid_resolution(ta_)

    po_ = po___.reshape(co_po_di.shape[0])

    if pl:

        grid_plot(co_po_di, po_, di_=di_, nu="Posterior Probability")

    if isnan(ta):

        return co_po_di, po_

    else:

        taco_ = unique(ta_)

        ie = absolute(taco_ - ta).argmin()

        ie_ = s_[ie :: taco_.size]

        co_po_dino = co_po_di[ie_, :-1]

        pono_ = po_[ie_]

        if pl:

            grid_plot(
                co_po_dino,
                pono_,
                di_=di_[:-1],
                nu="P({} = {:.2e} (~{}) | {})".format(
                    di_[-1], taco_[ie], ta, *di_[:-1]
                ),
            )

        return co_po_dino, pono_
