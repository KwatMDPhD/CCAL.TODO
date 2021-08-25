from numpy import logical_or, quantile, sort

from .check_is_not_nan import check_is_not_nan


def check_is_extreme(ar, di, th_=(), n_ex=0, st=0.0):

    arn = ar[check_is_not_nan(ar)]

    if 0 < len(th_):

        lo, hi = th_

    elif 0 < n_ex:

        if n_ex < 1:

            lo = quantile(arn, n_ex)

            hi = quantile(arn, 1 - n_ex)

        else:

            arns = sort(arn, axis=None)

            lo = arns[n_ex - 1]

            hi = arns[-n_ex]

    elif 0 < st:

        me = arn.mean()

        st *= arn.std()

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
