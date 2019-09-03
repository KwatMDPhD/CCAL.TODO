from numpy import log


def compute_vector_entropy(vector):

    probability = vector / vector.sum()

    return -(probability * log(probability)).sum()
