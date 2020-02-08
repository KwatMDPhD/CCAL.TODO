from numpy import sqrt
from scipy.stats import norm


def get_moe(array, confidence=0.95):

    return norm.ppf(q=confidence) * array.std() / sqrt(array.size)
