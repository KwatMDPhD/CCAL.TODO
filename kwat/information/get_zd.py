from .get_kld import get_kld


def get_zd(ve1, ve2):

    kl1_ = get_kld(ve1, ve2)

    kl2_ = get_kld(ve2, ve1)

    return kl1_, kl2_, kl1_ - kl2_
