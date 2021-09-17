from .get_kld import get_kld


def get_zd(nu1_, nu2_):

    kl1_ = get_kld(nu1_, nu2_)

    kl2_ = get_kld(nu2_, nu1_)

    return kl1_, kl2_, kl1_ - kl2_
