from numpy import diff, full, isnan, log as loge, log2, log10, nan, nanmin, unique
from scipy.stats import rankdata


def error_nan(array):

    assert not isnan(array).any()


def is_sorted(array):

    error_nan(array)

    diff_ = diff(array)

    return (diff_ <= 0).all() or (0 <= diff_).all()


def guess_type(array):

    error_nan(array)

    if all(float(x).is_integer() for x in array.flatten()):

        n_unique = unique(array).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= 16:

            return "categorical"

    return "continuous"


def clip(array, standard_deviation):

    error_nan(array)

    assert 0 <= standard_deviation

    mean = array.mean()

    margin = array.std() * standard_deviation

    return array.clip(min=mean - margin, max=mean + margin)


def shift_minimum(array, minimum):

    error_nan(array)

    if minimum == "0<":

        minimum = array[0 < array].min()

    return array + minimum - nanmin(array)


def log(array, log_base=2):

    error_nan(array)

    assert (0 < array).all()

    if log_base in (2, "2"):

        log_ = log2

    elif log_base == "e":

        log_ = loge

    elif log_base in (10, "10"):

        log_ = log10

    return log_(array)


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


def ignore_nan_and_function_1(
    array, function, *function_arguments, update=False, **function_keyword_arguments
):

    is_not_nan = ~isnan(array)

    output = function(
        array[is_not_nan], *function_arguments, **function_keyword_arguments
    )

    if update:

        array_ = full(array.shape, nan)

        array_[is_not_nan] = output

        return array_

    else:

        return output


def ignore_nan_and_function_2(
    array_0, array_1, function, *function_arguments, **function_keyword_arguments,
):

    is_not_nan = ~isnan(array_0) & ~isnan(array_1)

    return function(
        array_0[is_not_nan],
        array_1[is_not_nan],
        *function_arguments,
        **function_keyword_arguments,
    )
