from numpy import isinf, isnan


def check_array_for_bad(array, raise_for_bad=True):

    bads = []

    is_nan = isnan(array)

    if is_nan.any():

        bads.append("nan")

    is_inf = isinf(array)

    if is_inf.any():

        bads.append("inf")

    is_bad = is_nan | is_inf

    n_bad = is_bad.sum()

    if 0 < n_bad and raise_for_bad:

        n_good = array.size - n_bad

        bads = "|".join(bads)

        raise ValueError(f"{n_good} good & {n_bad} bad ({bads}).")

    return is_bad
