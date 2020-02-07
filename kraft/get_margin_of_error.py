from numpy import sqrt
from scipy.stats import norm


def get_margin_of_error(array, confidence=0.95):

    return norm.ppf(q=confidence) * array.std() / sqrt(array.size)
