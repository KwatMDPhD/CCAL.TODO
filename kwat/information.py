from numpy import absolute, array, exp, log, nan, outer, sign, sqrt, unique
from scipy.stats import pearsonr

from .array import normalize
from .density import get_bandwidth
from .grid import make_1d_grid
from .probability import get_probability


def get_entropy(ve):

    pr_ = ve / ve.sum()

    return -(pr_ * log(pr_)).sum()


def get_kld(ve1, ve2):

    return ve1 * log(ve1 / ve2)


def get_jsd(ve1, ve2, nu3_=None):

    if nu3_ is None:

        nu3_ = (ve1 + ve2) / 2

    kl1_ = get_kld(ve1, nu3_)

    kl2_ = get_kld(ve2, nu3_)

    return kl1_, kl2_, kl1_ - kl2_


def get_zd(ve1, ve2):

    kl1_ = get_kld(ve1, ve2)

    kl2_ = get_kld(ve2, ve1)

    return kl1_, kl2_, kl1_ - kl2_


def get_ic(ve1, ve2):

    if 1 in [unique(ve1).size, unique(ve2).size]:

        return nan

    ve1 = normalize(ve1, "-0-")

    ve2 = normalize(ve2, "-0-")

    pe = pearsonr(ve1, ve2)[0]

    ex = 0.1

    n_co = 24

    co1_ = make_1d_grid(ve1.min(), ve1.max(), ex, n_co)

    co2_ = make_1d_grid(ve2.min(), ve2.max(), ex, n_co)

    fa = 1 - abs(pe) * 2 / 3

    ba1 = get_bandwidth(ve1) * fa

    ba2 = get_bandwidth(ve2) * fa
    print(ba1, ba2)

    pr_ = get_probability(
        array([ve1, ve2]).T,
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


def get_icd(ve1, ve2):

    return (-get_ic(ve1, ve2) + 1) / 2


def get_signal_to_noise(ve0, ve1):

    me0 = ve0.mean()

    me1 = ve1.mean()

    st0 = ve0.std()

    st1 = ve1.std()

    lo0 = 0.2 * absolute(me0)

    lo1 = 0.2 * absolute(me1)

    if me0 == 0:

        st0 = 0.2

    elif st0 < lo0:

        st0 = lo0

    if me1 == 0:

        st1 = 0.2

    elif st1 < lo1:

        st1 = lo1

    return (me1 - me0) / (st0 + st1)
