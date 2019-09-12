from numpy import isinf, isnan


def check_array_for_bad(array, raise_for_bad=True):

    bad_kinds = []

    is_nan = isnan(array)

    if is_nan.any():

        bad_kinds.append("nan")

    is_inf = isinf(array)

    if is_inf.any():

        bad_kinds.append("inf")

    is_bad = is_nan | is_inf

    if is_bad.any() and raise_for_bad:

        raise

    return is_bad
