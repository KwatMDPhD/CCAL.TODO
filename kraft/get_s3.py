from .FLOAT_RESOLUTION import FLOAT_RESOLUTION
from .get_kl import get_kl


def get_s3(vector_0, vector_1):

    vector_0 += FLOAT_RESOLUTION

    vector_1 += FLOAT_RESOLUTION

    vector_reference = (vector_0 + vector_1) / 2

    kl_0 = get_kl(vector_0, vector_reference)

    kl_1 = get_kl(vector_1, vector_reference)

    return kl_0, kl_1, kl_0 - kl_1
