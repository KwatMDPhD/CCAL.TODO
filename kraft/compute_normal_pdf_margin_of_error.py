from numpy import nan, sqrt
from scipy.stats import norm

from .check_array_for_bad import check_array_for_bad


def compute_normal_pdf_margin_of_error(array, confidence=0.95, raise_for_bad=True):

    is_good = ~check_array_for_bad(array, raise_for_bad=raise_for_bad)

    if is_good.any():

        array_good = array[is_good]

        return norm.ppf(q=confidence) * array_good.std() / sqrt(array_good.size)

    else:

        return nan
