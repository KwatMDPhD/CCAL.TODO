from numpy import log


def compute_1d_array_entropy(_1d_array):

    probabilities = _1d_array / _1d_array.sum()

    return -(probabilities * log(probabilities)).sum()
