from numpy import asarray, exp, log, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .grid import make_1d_grid
from .kernel_density import get_bandwidth
from .number___ import normalize
from .probability import get_probability


def get_entropy(nu_):

    pr_ = nu_ / nu_.sum()

    return -(pr_ * log(pr_)).sum()


def get_kld(nu1_, nu2_):

    return nu1_ * log(nu1_ / nu2_)


def get_jsd(nu1_, nu2_, nu3_=None):

    if nu3_ is None:

        nu3_ = (nu1_ + nu2_) / 2

    kl1_ = get_kld(nu1_, nu3_)

    kl2_ = get_kld(nu2_, nu3_)

    return kl1_, kl2_, kl1_ - kl2_


def get_zd(nu1_, nu2_):

    kl1_ = get_kld(nu1_, nu2_)

    kl2_ = get_kld(nu2_, nu1_)

    return kl1_, kl2_, kl1_ - kl2_


def get_ic(nu1_, nu2_):

    if 1 in [unique(nu1_).size, unique(nu2_).size]:

        return nan

    nu1_ = normalize(nu1_, "-0-")

    nu2_ = normalize(nu2_, "-0-")

    pe = pearsonr(nu1_, nu2_)[0]

    ex = 0.1

    n_co = 24

    co1_ = make_1d_grid(nu1_.min(), nu1_.max(), ex, n_co)

    co2_ = make_1d_grid(nu2_.min(), nu2_.max(), ex, n_co)

    fa = 1 - abs(pe) * 2 / 3

    ba1 = get_bandwidth(nu1_) * fa

    ba2 = get_bandwidth(nu2_) * fa

    pr_ = get_probability(
        asarray([nu1_, nu2_]).T,
        ba_=[ba1, ba2],
        co__=[co1_, co2_],
        pl=False,
    )[1]

    pr_di_di = pr_.reshape([n_co, n_co])

    re1 = co1_[1] - co1_[0]

    re2 = co2_[1] - co2_[0]

    pr1 = pr_di_di.sum(axis=1) * re2

    pr2 = pr_di_di.sum(axis=0) * re1

    jo_ = outer(pr1, pr2).ravel()

    mu = get_kld(pr_, jo_).sum() * re1 * re2

    return sqrt(1 - exp(-2 * mu)) * sign(pe)


def get_icd(nu1_, nu2_):

    return (-get_ic(nu1_, nu2_) + 1) / 2
