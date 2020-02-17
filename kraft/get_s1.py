from .get_kl import get_kl


def get_s1(vector_0, vector_1, vector_reference, weight_0=1, weight_1=1):

    kl_0 = get_kl(vector_0, vector_reference) * weight_0

    kl_1 = get_kl(vector_1, vector_reference) * weight_1

    return kl_0, kl_1, kl_0 - kl_1
