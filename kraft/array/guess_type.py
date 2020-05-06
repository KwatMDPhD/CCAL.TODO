from numpy import unique

from .error_nan import error_nan


def guess_type(array):

    error_nan(array)

    array_flat = array.flatten()

    if all(float(number).is_integer() for number in array_flat):

        n_unique = unique(array_flat).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= 16:

            return "categorical"

    return "continuous"
