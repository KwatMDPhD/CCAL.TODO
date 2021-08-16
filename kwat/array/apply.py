from numpy import full, nan
from .check_is_not_nan import check_is_not_nan


def apply(ar, fu, *ar_, up=False, **ke_):

    arbo = check_is_not_nan(ar)

    re = fu(ar[arbo], *ar_, **ke_)

    if up:

        ar2 = full(ar.shape, nan)

        ar2[arbo] = re

        return ar2

    return re
