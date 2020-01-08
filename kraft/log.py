from numpy import log as loge, log2, log10


def log_array(array, min_before_logging=None, log_base="e"):

    if min_before_logging is not None:

        if min_before_logging == "0<":

            min_before_logging = array[0 < array].min()

        array += min_before_logging - array.min()

    if log_base == "2":

        log_ = log2

    elif log_base == "e":

        log_ = loge

    elif log_base == "10":

        log_ = log10

    return log_(array)
