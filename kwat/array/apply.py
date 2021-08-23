from numpy import full, nan

from .check_is_not_nan import check_is_not_nan


def apply(ar, fu, *arg, up=False, **ke):

    arn = check_is_not_nan(ar)

    re = fu(ar[arn], *arg, **ke)

    if up:

        arr = full(ar.shape, nan)

        arr[arn] = re

        return arr

    return re
