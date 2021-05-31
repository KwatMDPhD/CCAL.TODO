from numpy import (
    full,
    integer,
    isnan,
    log as loge,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    nan,
    quantile,
    sort,
    unique,
)
from scipy.stats import rankdata


def check_is_not_nan(nu___):

    return logical_not(isnan(nu___))


def get_not_nan_unique(nu___):

    return unique(nu___[check_is_not_nan(nu___)])


def clip(nu___, st):

    me = nu___.mean()

    st *= nu___.std()

    return nu___.clip(me - st, me + st)


def normalize(nu___, me, ra="average"):

    if me == "-0-":

        return (nu___ - nu___.mean()) / nu___.std()

    elif me == "0-1":

        mi = nu___.min()

        return (nu___ - mi) / (nu___.max() - mi)

    elif me == "sum":

        return nu___ / nu___.sum()

    elif me == "rank":

        return rankdata(nu___, ra).reshape(nu___.shape)


def shift_minimum(nu___, mi):

    if mi == "0<":

        bo___ = 0 < nu___

        if bo___.any():

            mi = nu___[bo___].min()

        else:

            mi = 1

        print("Shifting the minimum to {}...".format(mi))

    return nu___ + mi - nu___.min()


def log(nu___, ba=2):

    return {2: log2, "e": loge, 10: log10,}[
        ba
    ](nu___)


def guess_type(nu___, ma=16):

    if all(isinstance(nu, integer) for nu in nu___.ravel()):

        n_ca = unique(nu___).size

        if n_ca <= 2:

            return "binary"

        elif n_ca <= ma:

            return "categorical"

    return "continuous"


def check_is_extreme(nu___, di, th_=(), n_ex=0, st=0.0):

    nuno___ = nu___[check_is_not_nan(nu___)]

    if 0 < len(th_):

        lo, hi = th_

    elif 0 < n_ex:

        if n_ex < 1:

            lo = quantile(nuno___, n_ex)

            hi = quantile(nuno___, 1 - n_ex)

        else:

            nuno___ = sort(nuno___, None)

            lo = nuno___[n_ex - 1]

            hi = nuno___[-n_ex]

    elif 0 < st:

        me = nuno___.mean()

        st *= nuno___.std()

        lo = me - st

        hi = me + st

    if di == "<>":

        return logical_or(nu___ <= lo, hi <= nu___)

    elif di == "<":

        return nu___ <= lo

    elif di == ">":

        return hi <= nu___


def apply_on_1(nu___, fu, *ar_, up=False, **ke_):

    bo___ = check_is_not_nan(nu___)

    re = fu(nu___[bo___], *ar_, **ke_)

    if up:

        nu2___ = full(nu___.shape, nan)

        nu2___[bo___] = re

        return nu2___

    return re


def apply_on_2(nu1___, nu2___, fu, *ar_, **ke_):

    bo___ = logical_and(check_is_not_nan(nu1___), check_is_not_nan(nu2___))

    return fu(nu1___[bo___], nu2___[bo___], *ar_, **ke_)


def apply_along_on_2(nu1___, nu2___, fu, *ar_, **ke_):

    n_ro1 = nu1___.shape[0]

    n_ro2 = nu2___.shape[0]

    nu3___ = full([n_ro1, n_ro2], nan)

    for ie1 in range(n_ro1):

        nu1_ = nu1___[ie1]

        for ie2 in range(n_ro2):

            nu2_ = nu2___[ie2]

            nu3___[ie1, ie2] = fu(nu1_, nu2_, *ar_, **ke_)

    return nu3___
