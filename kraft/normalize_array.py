from numpy import apply_along_axis, full, nan
from scipy.stats import rankdata

from .check_array_for_bad import check_array_for_bad


def _normalize_array(array, method, rank_method, raise_for_bad):

    is_good = ~check_array_for_bad(array, raise_for_bad=raise_for_bad)

    array_normalized = full(array.shape, nan)

    if is_good.any():

        array_good = array[is_good]

        if method == "-0-":

            array_good_std = array_good.std()

            if array_good_std == 0:

                array_normalized[is_good] = 0

            else:

                array_normalized[is_good] = (
                    array_good - array_good.mean()
                ) / array_good_std

        elif method == "0-1":

            array_good_min = array_good.min()

            array_good_range = array_good.max() - array_good_min

            if array_good_range == 0:

                array_normalized[is_good] = nan

            else:

                array_normalized[is_good] = (
                    array_good - array_good_min
                ) / array_good_range

        elif method == "sum":

            if array_good.min() < 0:

                raise ValueError("Sum normalize only positives.")

            else:

                array_good_sum = array_good.sum()

                if array_good_sum == 0:

                    array_normalized[is_good] = 1 / is_good.sum()

                else:

                    array_normalized[is_good] = array_good / array_good_sum

        elif method == "rank":

            array_normalized[is_good] = rankdata(array_good, method=rank_method)

    return array_normalized


def normalize_array(
    array, axis, method, rank_method="average", raise_for_bad=True
):

    if axis is None:

        return _normalize_array(array, method, rank_method, raise_for_bad)

    else:

        return apply_along_axis(
            _normalize_array, axis, array, method, rank_method, raise_for_bad
        )
