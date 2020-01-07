from numpy import sqrt
from scipy.stats import norm


def compute_margin_of_error(values, confidence=0.95):

    return norm.ppf(q=confidence) * values.std() / sqrt(values.size)
