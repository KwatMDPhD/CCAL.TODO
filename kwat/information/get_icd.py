from .get_ic import get_ic


def get_icd(ve1, ve2):
    return (-get_ic(ve1, ve2) + 1) / 2
