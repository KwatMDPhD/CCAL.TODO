from numpy import nanmin

from .error_nan import error_nan


def shift_minimum(array, minimum):

    error_nan(array)

    if minimum == "0<":

        minimum = array[0 < array].min()

    return array + minimum - nanmin(array)
