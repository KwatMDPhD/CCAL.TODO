from .compute_ic import compute_ic


def compute_icd(vector_0, vector_1):

    return -(compute_ic(vector_0, vector_1) - 1)
