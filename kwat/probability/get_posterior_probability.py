from numpy import absolute, apply_along_axis, isnan, nan, s_, unique

from ..grid import get_1d_grid, get_1d_grid_resolution, plot
from .get_probability import get_probability


def _get_probability(nu___):

    return nu___ / nu___.sum()


def get_posterior_probability(nu_po_di, ta=nan, co__=(), pl=True, na_=(), **ke_ar):

    co_po_di, pr_ = get_probability(nu_po_di, co__=co__, pl=pl, na_=na_, **ke_ar)

    cot_ = co_po_di[:, -1]

    pr___ = pr_.reshape([co_.size for co_ in get_1d_grid(co_po_di)])

    po___ = apply_along_axis(_get_probability, -1, pr___) * get_1d_grid_resolution(cot_)

    po_ = po___.reshape(co_po_di.shape[0])

    if pl:

        plot(co_po_di, po_, na_=na_ + ["Posterior Probability"])

    if isnan(ta):

        return co_po_di, po_

    else:

        cotu_ = unique(cot_)

        ie = absolute(cotu_ - ta).argmin()

        ie_ = s_[ie :: cotu_.size]

        co_po_dit = co_po_di[ie_, :-1]

        pot_ = po_[ie_]

        if pl:

            plot(
                co_po_dit,
                pot_,
                na_=na_[:-1]
                + [
                    "P({} = {:.2e} (~{}) | {})".format(
                        na_[-1], cotu_[ie], ta, *na_[:-1]
                    )
                ],
            )

        return co_po_dit, pot_
