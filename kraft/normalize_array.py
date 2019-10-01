from numpy import full, nan
from scipy.stats import rankdata

from .check_array_for_bad import check_array_for_bad


def normalize_array(array, method, rank_method="average", raise_for_bad=True):

    is_good = ~check_array_for_bad(array, raise_for_bad=raise_for_bad)

    array_ = full(array.shape, nan)

    if not is_good.any():

        return array_

    array_good = array[is_good]

    if method == "-0-":

        array_[is_good] = (array_good - array_good.mean()) / array_good.std()

    elif method == "0-1":

        array_good_min = array_good.min()

        array_[is_good] = (array_good - array_good_min) / (
            array_good.max() - array_good_min
        )

    elif method == "sum":

        assert 0 <= array_good.min()

        array_[is_good] = array_good / array_good.sum()

    elif method == "rank":

        array_[is_good] = rankdata(array_good, method=rank_method)

    return array_
