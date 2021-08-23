from numpy import logical_and

from ..array import check_is_not_nan


def ignore_nan_and_apply(ar1, ar2, fu):

    bo = logical_and(check_is_not_nan(ar1), check_is_not_nan(ar2))

    return fu(ar1[bo], ar2[bo])
