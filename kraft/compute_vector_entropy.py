from numpy import log


def compute_vector_entropy(vector):

    probabilities = vector / vector.sum()

    return -(probabilities * log(probabilities)).sum()
