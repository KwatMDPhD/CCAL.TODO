from .get_kld import get_kld


def get_jsd(vector_0, vector_1, vector_reference=None):

    if vector_reference is None:

        vector_reference = (vector_0 + vector_1) / 2

    kld_0 = get_kld(vector_0, vector_reference)

    kld_1 = get_kld(vector_1, vector_reference)

    return kld_0, kld_1, kld_0 - kld_1
