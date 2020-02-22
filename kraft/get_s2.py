from .FLOAT_RESOLUTION import FLOAT_RESOLUTION
from .get_kl import get_kl


def get_s2(vector_0, vector_1):

    vector_0 += FLOAT_RESOLUTION

    vector_1 += FLOAT_RESOLUTION

    kl_0 = get_kl(vector_0, vector_1)

    kl_1 = get_kl(vector_1, vector_0)

    return kl_0, kl_1, kl_0 - kl_1
