from .get_kld import get_kld


def get_jsd(ve1, ve2, nu3_=None):

    if nu3_ is None:

        nu3_ = (ve1 + ve2) / 2

    kl1_ = get_kld(ve1, nu3_)

    kl2_ = get_kld(ve2, nu3_)

    return kl1_, kl2_, kl1_ - kl2_
