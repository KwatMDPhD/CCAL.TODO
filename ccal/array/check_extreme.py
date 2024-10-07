from numpy import logical_or, quantile, sort

from .check_not_nan import check_not_nan


def check_extreme(nu___, di, th_=(), n_ex=0, st=0.0):
    nug___ = nu___[check_not_nan(nu___)]

    if 0 < len(th_):
        lo, hi = th_

    elif 0 < n_ex:
        if n_ex < 1:
            lo = quantile(nug___, n_ex)

            hi = quantile(nug___, 1 - n_ex)

        else:
            nug_ = sort(nug___, axis=None)

            lo = nug_[n_ex - 1]

            hi = nug_[-n_ex]

    elif 0 < st:
        me = nug___.mean()

        st *= nug___.std()

        lo = me - st

        hi = me + st

    if "<" in di:
        lo___ = nu___ <= lo

    if ">" in di:
        hi___ = hi <= nu___

    if di == "<>":
        return logical_or(lo___, hi___)

    elif di == "<":
        return lo___

    elif di == ">":
        return hi___
