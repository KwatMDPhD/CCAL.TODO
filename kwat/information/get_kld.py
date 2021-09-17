from ..array import log


def get_kld(nu1_, nu2_):

    return nu1_ * log(nu1_ / nu2_)
