from KDEpy import FFTKDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .CONSTANT import FLOAT_RESOLUTION
from .grid import make_1d_grid, make_nd_grid, plot as grid_plot


def get_bandwidth(nu_):

    return KDEMultivariate(nu_, "c").bw[0]


def get_density(nu_po_di, ba_=(), co__=(), pl=True, di_=()):

    nu_di_po = nu_po_di.T

    n_di = nu_di_po.shape[0]

    if len(ba_) != n_di:

        ba_ = [get_bandwidth(nu_) for nu_ in nu_di_po]

    if len(co__) != n_di:

        co__ = [make_1d_grid(nu_.min(), nu_.max(), 0.1, 8) for nu_ in nu_di_po]

    co_po_di = make_nd_grid(co__)

    de_ = FFTKDE(bw=ba_).fit(nu_po_di).evaluate(co_po_di).clip(FLOAT_RESOLUTION)

    if pl:

        grid_plot(co_po_di, de_, di_=di_, nu="Density")

    return co_po_di, de_
