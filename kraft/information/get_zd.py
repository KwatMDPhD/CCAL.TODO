from .FLOAT_RESOLUTION import FLOAT_RESOLUTION
from .get_kld import get_kld


def get_zd(vector_0, vector_1):

    vector_0 += FLOAT_RESOLUTION

    vector_1 += FLOAT_RESOLUTION

    kld_0 = get_kld(vector_0, vector_1)

    kld_1 = get_kld(vector_1, vector_0)

    return kld_0, kld_1, kld_0 - kld_1
