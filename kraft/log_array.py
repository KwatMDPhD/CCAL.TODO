from numpy import log as loge, log2, log10


def log_array(
    array, shift_as_necessary_to_achieve_min_before_logging=None, log_base="e",
):

    if shift_as_necessary_to_achieve_min_before_logging is not None:

        if shift_as_necessary_to_achieve_min_before_logging == "0<":

            shift_as_necessary_to_achieve_min_before_logging = array[0 < array].min()

        array_good_min = array.min()

        if array_good_min < shift_as_necessary_to_achieve_min_before_logging:

            array += shift_as_necessary_to_achieve_min_before_logging - array_good_min

    if str(log_base) == "2":

        log_ = log2

    elif log_base == "e":

        log_ = loge

    elif str(log_base) == "10":

        log_ = log10

    return log_(array)
