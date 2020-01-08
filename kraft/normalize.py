from scipy.stats import rankdata


def normalize_array(array, method, rank_method="average"):

    if method == "-0-":

        array_good_std = array.std()

        assert array_good_std != 0

        return (array - array.mean()) / array_good_std

    elif method == "0-1":

        array_good_min = array.min()

        array_good_range = array.max() - array_good_min

        assert array_good_range != 0

        return (array - array_good_min) / array_good_range

    elif method == "sum":

        assert 0 <= array.min()

        array_good_sum = array.sum()

        assert array_good_sum != 0

        return array / array_good_sum

    elif method == "rank":

        return rankdata(array, method=rank_method)
