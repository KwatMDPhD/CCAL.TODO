from numpy import full, logical_and, nan, where

from .array import check_is_not_nan


def apply(ar1, ar2, fu, *ar_, **ke_):

    arbo = logical_and(check_is_not_nan(ar1), check_is_not_nan(ar2))

    return fu(ar1[arbo], ar2[arbo], *ar_, **ke_)


def apply_along(ar1, ar2, fu, *ar_, **ke_):

    n_ro1 = ar1.shape[0]

    n_ro2 = ar2.shape[0]

    ar3 = full([n_ro1, n_ro2], nan)

    for ie1 in range(n_ro1):

        nu1_ = ar1[ie1]

        for ie2 in range(n_ro2):

            nu2_ = ar2[ie2]

            ar3[ie1, ie2] = fu(nu1_, nu2_, *ar_, **ke_)

    return ar3


def separate(arb, ar):

    return ar[where(arb == 0)[0]], ar[where(arb == 1)[0]]
