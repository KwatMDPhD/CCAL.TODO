from .compute_ic import compute_ic


def compute_icd(vector_0, vector_1):

    return (1 - compute_ic(vector_0, vector_1)) / 2
