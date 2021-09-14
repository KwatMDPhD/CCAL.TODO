from numpy import logical_not

from .apply import apply


def separate_and_apply(bi___, nu___, fu):

    bo0___ = bi___ == 0

    bo1___ = logical_not(bo0___)

    return apply(nu___[bo0___], nu___[bo1___], fu)
