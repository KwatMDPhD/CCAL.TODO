from numpy import isnan, unique


def guess_type(array):

    array_flat = array.flatten()

    array_flat_not_nan = array_flat[~isnan(array_flat)]

    if all(float(number).is_integer() for number in array_flat_not_nan):

        n_unique = unique(array_flat_not_nan).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= 16:

            return "categorical"

    return "continuous"
