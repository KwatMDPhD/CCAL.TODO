from numpy import array, exp, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from ..array import normalize
from ..grid import make_1d_grid
from ..probability import get_probability
from .get_kld import get_kld


def get_ic(nu1_, nu2_):

    if 1 in [unique(nu1_).size, unique(nu2_).size]:

        return nan

    nu1_ = normalize(nu1_, "-0-")

    nu2_ = normalize(nu2_, "-0-")

    ex = 1 / 3

    n_co = 24

    nu_di_di = array([nu1_, nu2_]).T

    co1_ = make_1d_grid(nu1_.min(), nu1_.max(), ex, n_co)

    co2_ = make_1d_grid(nu2_.min(), nu2_.max(), ex, n_co)

    pe = pearsonr(nu1_, nu2_)[0]

    fa = 1 - abs(pe) * 2 / 3

    pr_ = get_probability(
        nu_di_di,
        co__=[co1_, co2_],
        pl=False,
        bw=fa,
    )[1]

    pr_di_di = pr_.reshape([n_co] * 2)

    re1 = co1_[1] - co1_[0]

    re2 = co2_[1] - co2_[0]

    pr1 = pr_di_di.sum(axis=1) * re2

    pr2 = pr_di_di.sum(axis=0) * re1

    jo_ = outer(pr1, pr2).ravel()

    mu = get_kld(pr_, jo_).sum() * re1 * re2

    return sqrt(1 - exp(-2 * mu)) * sign(pe)
