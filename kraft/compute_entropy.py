from numpy import log


def compute_entropy(vector):

    assert (0 < vector).all()

    probability = vector / vector.sum()

    return -(probability * log(probability)).sum()
