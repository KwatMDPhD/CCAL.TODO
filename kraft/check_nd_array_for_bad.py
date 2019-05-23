from numpy import isinf, isnan


def check_nd_array_for_bad(nd_array, raise_for_bad=True):

    bads = []

    is_nan = isnan(nd_array)

    if is_nan.any():

        bads.append("nan")

    is_inf = isinf(nd_array)

    if is_inf.any():

        bads.append("inf")

    is_bad = is_nan | is_inf

    n_bad = is_bad.sum()

    if raise_for_bad and n_bad:

        raise ValueError(
            f"{nd_array.size - n_bad} good & {n_bad} bad ({'|'.join(bads)})."
        )

    else:

        return is_bad
