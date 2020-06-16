from scipy.stats import rankdata

from .error_nan import error_nan


def normalize(array, method, rank_method="average"):

    error_nan(array)

    if method == "-0-":

        standard_deviation = array.std()

        assert not standard_deviation == 0

        return (array - array.mean()) / standard_deviation

    elif method == "0-1":

        min_ = array.min()

        range_ = array.max() - min_

        assert not range_ == 0

        return (array - min_) / range_

    elif method == "sum":

        assert (0 <= array).all()

        sum_ = array.sum()

        assert not sum_ == 0

        return array / sum_

    elif method == "rank":

        return rankdata(array, method=rank_method)
