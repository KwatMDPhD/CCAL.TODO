from .get_kld import get_kld


def get_zd(ve1, ve2):

    kl1 = get_kld(ve1, ve2)

    kl2 = get_kld(ve2, ve1)

    return kl1, kl2, kl1 - kl2
