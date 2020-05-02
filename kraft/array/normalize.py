from scipy.stats import rankdata


def normalize(array, method, rank_method="average"):

    if method == "-0-":

        return (array - array.mean()) / array.std()

    elif method == "0-1":

        min_ = array.min()

        return (array - min_) / (array.max() - min_)

    elif method == "sum":

        return array / array.sum()

    elif method == "rank":

        return rankdata(array, method=rank_method)
