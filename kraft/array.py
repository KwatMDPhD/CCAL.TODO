from numpy import diff, isnan, log as loge, log2, log10, nan, nanmin, unique
from numpy.random import seed, shuffle as numpy_shuffle
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED


def error_nan(array):

    assert not isnan(array).any()


def guess_type(array):

    error_nan(array)

    if all(float(number).is_integer() for number in array.flatten()):

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

        log_function = log2

    elif log_base == "e":

        log_function = loge

    elif log_base in (10, "10"):

        log_function = log10

    return log_function(array)


def normalize(array, method, rank_method="average"):

    error_nan(array)

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

        return rankdata(array, method=rank_method).reshape(array.shape)


def ignore_nan_and_function_1(
    array, function, *function_arguments, update=False, **function_keyword_arguments
):

    is_good = ~isnan(array)

    if not is_good.any():

        return nan

    returned = function(
        array[is_good], *function_arguments, **function_keyword_arguments
    )

    if update:

        array = array.copy()

        array[is_good] = returned

        return array

    else:

        return returned


def ignore_nan_and_function_2(
    array0, array1, function, *function_arguments, **function_keyword_arguments,
):

    is_good = ~isnan(array0) & ~isnan(array1)

    if not is_good.any():

        return nan

    return function(
        array0[is_good],
        array1[is_good],
        *function_arguments,
        **function_keyword_arguments,
    )


def check_is_sorted(vector):

    assert vector.ndim == 1

    error_nan(vector)

    differences = diff(vector)

    return (differences <= 0).all() or (0 <= differences).all()


def shuffle(matrix, axis, random_seed=RANDOM_SEED):

    assert matrix.ndim == 2

    error_nan(matrix)

    matrix = matrix.copy()

    seed(seed=random_seed)

    if axis == 0:

        for i in range(matrix.shape[1]):

            numpy_shuffle(matrix[:, i])

    elif axis == 1:

        for i in range(matrix.shape[0]):

            numpy_shuffle(matrix[i, :])

    return matrix
