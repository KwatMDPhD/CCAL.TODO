from numpy import logical_and

from ..array import check_not_nan


def apply(ar1, ar2, fu):

    arn = logical_and(check_not_nan(ar1), check_not_nan(ar2))

    return fu(ar1[arn], ar2[arn])
