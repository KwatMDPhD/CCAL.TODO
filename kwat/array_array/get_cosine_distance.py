from numpy import dot
from numpy.linalg import norm


def get_cosine_distance(ve1, ve2):

    return dot(ve1, ve2) / (norm(ve1) * norm(ve2))
