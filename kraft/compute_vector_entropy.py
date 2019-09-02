from numpy import log


def compute_vector_entropy(_vector):

    probabilities = _vector / _vector.sum()

    return -(probabilities * log(probabilities)).sum()
