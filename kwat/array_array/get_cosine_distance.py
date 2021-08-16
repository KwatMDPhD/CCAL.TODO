from numpy import dot
from numpy.linalg import norm


def get_cosine_distance(ve0, ve1):

    return dot(ve0, ve1) / (norm(ve0) * norm(ve1))
