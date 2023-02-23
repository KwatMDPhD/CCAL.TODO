from ..array import log


def get_kld(ve1, ve2):
    return ve1 * log(ve1 / ve2)
