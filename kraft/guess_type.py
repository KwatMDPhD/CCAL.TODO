from numpy import asarray, isnan, unique


def guess_type(numbers):

    numbers_flat = asarray(numbers).flatten()

    numbers_flat_not_nan = numbers_flat[~isnan(numbers_flat)]

    if all(float(number).is_integer() for number in numbers_flat_not_nan):

        n_unique = unique(numbers_flat_not_nan).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= 16:

            return "categorical"

    return "continuous"
