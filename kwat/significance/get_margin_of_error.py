from numpy import sqrt
from scipy.stats import norm


def get_margin_of_error(ve, co=0.95):
    return norm.ppf(co) * ve.std() / sqrt(ve.size)
