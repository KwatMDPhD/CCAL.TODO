from .get_ic import get_ic


def get_icd(nu1_, nu2_):

    return (-get_ic(nu1_, nu2_) + 1) / 2
