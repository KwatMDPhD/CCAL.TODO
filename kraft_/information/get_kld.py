from numpy import log


def get_kld(vector_0, vector_1):

    return vector_0 * log(vector_0 / vector_1)
