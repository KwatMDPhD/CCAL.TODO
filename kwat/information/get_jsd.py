from .get_kld import get_kld


def get_jsd(ve1, ve2, ve3=None):
    if ve3 is None:
        ve3 = (ve1 + ve2) / 2

    kl1_ = get_kld(ve1, ve3)

    kl2_ = get_kld(ve2, ve3)

    return kl1_, kl2_, kl1_ - kl2_
