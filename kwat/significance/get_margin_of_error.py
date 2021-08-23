from numpy import sqrt
from scipy.stats import norm


def get_margin_of_error(nu_, co=0.95):

    return norm.ppf(co) * nu_.std() / sqrt(nu_.size)
