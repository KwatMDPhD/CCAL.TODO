from numpy import logical_not

from .apply import apply


def separate_and_apply(bi___, nu___, fu):

    bi0___ = bi___ == 0

    bi1___ = logical_not(bi0___)

    return apply(nu___[bi0___], nu___[bi1___], fu)
