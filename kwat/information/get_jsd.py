from .get_kld import get_kld


def get_jsd(nu1_, nu2_, nu3_=None):

    if nu3_ is None:

        nu3_ = (nu1_ + nu2_) / 2

    kl1_ = get_kld(nu1_, nu3_)

    kl2_ = get_kld(nu2_, nu3_)

    return kl1_, kl2_, kl1_ - kl2_
