from numpy import log


def get_entropy(vector):

    probability = vector / vector.sum()

    return -(probability * log(probability)).sum()
