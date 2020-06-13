from numpy import nanmin

from .error_nan import error_nan


def set_minimum(array, minimum):

    error_nan(array)

    if minimum == "0<":

        minimum = array[0 < array].min()

    else:

        array += min_before_logging - nanmin(array)

    return
