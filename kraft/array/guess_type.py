from numpy import unique

from .error_nan import error_nan


def guess_type(array):

    error_nan(array)

    if all(float(x).is_integer() for x in array.flatten()):

        n_unique = unique(array).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= 16:

            return "categorical"

    return "continuous"
