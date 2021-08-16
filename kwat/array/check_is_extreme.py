from numpy import logical_or, quantile, sort

from .check_is_not_nan import check_is_not_nan


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

    else:

        raise

    if di == "<>":

        return logical_or(ar <= lo, hi <= ar)

    elif di == "<":

        return ar <= lo

    elif di == ">":

        return hi <= ar
