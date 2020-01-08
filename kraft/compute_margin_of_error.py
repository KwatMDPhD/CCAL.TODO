from numpy import sqrt
from scipy.stats import norm


def compute_margin_of_error(vector, confidence=0.95):

    return norm.ppf(q=confidence) * vector.std() / sqrt(vector.size)
