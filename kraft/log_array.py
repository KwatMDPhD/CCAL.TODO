from numpy import full
from numpy import log as loge
from numpy import log2, log10, nan

from .check_array_for_bad import check_array_for_bad


def log_array(
    array,
    shift_as_necessary_to_achieve_min_before_logging=None,
    log_base="e",
    raise_for_bad=True,
):

    is_good = ~check_array_for_bad(array, raise_for_bad=raise_for_bad)

    array_ = full(array.shape, nan)

    if not is_good.any():

        return array_

    array_good = array[is_good]

    if shift_as_necessary_to_achieve_min_before_logging is not None:

        if shift_as_necessary_to_achieve_min_before_logging == "0<":

            shift_as_necessary_to_achieve_min_before_logging = array_good[
                0 < array_good
            ].min()

        min_ = array_good.min()

        if min_ < shift_as_necessary_to_achieve_min_before_logging:

            array_good += shift_as_necessary_to_achieve_min_before_logging - min_

    if str(log_base) == "2":

        log_ = log2

    elif log_base == "e":

        log_ = loge

    elif str(log_base) == "10":

        log_ = log10

    array_[is_good] = log_(array_good)

    return array_
