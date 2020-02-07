from .get_ic import get_ic


def get_icd(vector_0, vector_1):

    return -(get_ic(vector_0, vector_1) - 1)
