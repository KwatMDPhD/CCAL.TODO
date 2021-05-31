from numpy import (
    full,
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


def check_is_not_nan(ar):

    return logical_not(isnan(ar))


def get_not_nan_unique(ar):

    return unique(ar[check_is_not_nan(ar)])


def clip(ar, st):

    me = ar.mean()

    st *= ar.std()

    return ar.clip(me - st, me + st)


def normalize(ar, me, ra="average"):

    if me == "-0-":

        return (ar - ar.mean()) / ar.std()

    elif me == "0-1":

        mi = ar.min()

        return (ar - mi) / (ar.max() - mi)

    elif me == "sum":

        return ar / ar.sum()

    elif me == "rank":

        return rankdata(ar, ra).reshape(ar.shape)


def shift(ar, mi):

    if mi == "0<":

        arbo = 0 < ar

        if arbo.any():

            mi = ar[arbo].min()

        else:

            mi = 1

        print("Shifting the minimum to {}...".format(mi))

    return ar + mi - ar.min()


def log(ar, ba=2):

    return {2: log2, "e": loge, 10: log10,}[
        ba
    ](ar)


def guess_type(ar, ma=16):

    if all(float(nu).is_integer() for nu in ar.ravel()):

        n_ca = unique(ar).size

        if n_ca <= 2:

            return "binary"

        elif n_ca <= ma:

            return "categorical"

    return "continuous"


def check_is_extreme(ar, di, th_=(), n_ex=0, st=0.0):

    arno = ar[check_is_not_nan(ar)]

    if 0 < len(th_):

        lo, hi = th_

    elif 0 < n_ex:

        if n_ex < 1:

            lo = quantile(arno, n_ex)

            hi = quantile(arno, 1 - n_ex)

        else:

            arno = sort(arno, None)

            lo = arno[n_ex - 1]

            hi = arno[-n_ex]

    elif 0 < st:

        me = arno.mean()

        st *= arno.std()

        lo = me - st

        hi = me + st

    if di == "<>":

        return logical_or(ar <= lo, hi <= ar)

    elif di == "<":

        return ar <= lo

    elif di == ">":

        return hi <= ar


def apply_on_1(ar, fu, *ar_, up=False, **ke_):

    arbo = check_is_not_nan(ar)

    re = fu(ar[arbo], *ar_, **ke_)

    if up:

        ar2 = full(ar.shape, nan)

        ar2[arbo] = re

        return ar2

    return re


def apply_on_2(ar1, ar2, fu, *ar_, **ke_):

    arbo = logical_and(check_is_not_nan(ar1), check_is_not_nan(ar2))

    return fu(ar1[arbo], ar2[arbo], *ar_, **ke_)


def apply_along_on_2(ar1, ar2, fu, *ar_, **ke_):

    n_ro1 = ar1.shape[0]

    n_ro2 = ar2.shape[0]

    ar3 = full([n_ro1, n_ro2], nan)

    for ie1 in range(n_ro1):

        nu1_ = ar1[ie1]

        for ie2 in range(n_ro2):

            nu2_ = ar2[ie2]

            ar3[ie1, ie2] = fu(nu1_, nu2_, *ar_, **ke_)

    return ar3
