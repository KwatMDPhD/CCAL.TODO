from numpy import apply_along_axis

from .normalize_array import normalize_array


def normalize_array_on_axis(
    array, axis, method, rank_method="average", raise_for_bad=True
):

    return apply_along_axis(
        normalize_array, axis, array, method, rank_method, raise_for_bad
    )
