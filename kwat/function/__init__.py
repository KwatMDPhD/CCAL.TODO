from numpy import logical_and, where

from .array import check_is_not_nan


def separate_and_apply(ar, arb, fu):

    return fu(ar[where(arb == 0)[0]], ar[where(arb == 1)[0]])


def ignore_nan_and_apply(ar1, ar2, fu):

    bo = logical_and(check_is_not_nan(ar1), check_is_not_nan(ar2))

    return fu(ar1[bo], ar2[bo])
