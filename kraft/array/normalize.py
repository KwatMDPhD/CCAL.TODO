from scipy.stats import rankdata


def normalize(array, method, rank_method="average"):

    if method == "-0-":

        standard_deviation = array.std()

        assert standard_deviation != 0

        return (array - array.mean()) / standard_deviation

    elif method == "0-1":

        min_ = array.min()

        range_ = array.max() - min_

        assert range_ != 0

        return (array - min_) / range_

    elif method == "sum":

        assert (0 <= array).all()

        sum_ = array.sum()

        assert sum_ != 0

        return array / sum_

    elif method == "rank":

        return rankdata(array, method=rank_method)
