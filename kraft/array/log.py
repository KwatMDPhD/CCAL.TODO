from numpy import log as loge, log2, log10

from .error_nan import error_nan


def log(array, log_base=2):

    error_nan(array)

    assert (0 < array).all()

    if log_base in (2, "2"):

        log_ = log2

    elif log_base == "e":

        log_ = loge

    elif log_base in (10, "10"):

        log_ = log10

    return log_(array)
