from numpy import nan, sqrt
from scipy.stats import norm

from .check_array_for_bad import check_array_for_bad


def compute_normal_pdf_margin_of_error(pdf, confidence=0.95, raise_for_bad=True):

    is_good = ~check_array_for_bad(pdf, raise_for_bad=raise_for_bad)

    if not is_good.any():

        return nan

    pdf_good = pdf[is_good]

    return norm.ppf(q=confidence) * pdf_good.std() / sqrt(pdf_good.size)
