from numpy import (
    apply_along_axis,
    asarray,
    diff,
    full,
    integer,
    isnan,
    log as loge,
    log2,
    log10,
    logical_and,
    logical_not,
    logical_or,
    nan,
    quantile,
    sort,
    unique,
)
from numpy.random import seed, shuffle as shuffle_
from pandas import isna
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED

# ==============================================================================
# 1D array
# ==============================================================================


def check_is_in(_1d_array, lookup_1d_array):

    lookup = {value: None for value in lookup_1d_array}

    return asarray(tuple(value in lookup for value in _1d_array))


def map_int(_1d_array):

    value_to_integer = {}

    integer_to_value = {}

    integer = 0

    for value in _1d_array:

        if value not in value_to_integer:

            value_to_integer[value] = integer

            integer_to_value[integer] = value

            integer += 1

    return value_to_integer, integer_to_value


# ==============================================================================
# Vector
# ==============================================================================


def check_is_all_sorted(vector):

    difference_ = diff(vector)

    return (difference_ <= 0).all() or (0 <= difference_).all()


def check_is_extreme(
    vector, direction, low_and_high=None, n=None, standard_deviation=None
):

    vector2 = vector[~isnan(vector)]

    if low_and_high is not None:

        low, high = low_and_high

    elif n is not None:

        if n < 1:

            low = quantile(vector2, n)

            high = quantile(vector2, 1 - n)

        else:

            vector_sort = sort(vector2)

            low = vector_sort[n - 1]

            high = vector_sort[-n]

    elif standard_deviation is not None:

        mean = vector2.mean()

        margin = vector2.std() * standard_deviation

        low = mean - margin

        high = mean + margin

    if direction == "<>":

        return logical_or(vector <= low, high <= vector)

    elif direction == "<":

        return vector <= low

    elif direction == ">":

        return high <= vector


# ==============================================================================
# 2D array
# ==============================================================================


def shuffle(_2d_array, random_seed=RANDOM_SEED):

    _2d_array_copy = _2d_array.copy()

    seed(seed=random_seed)

    for index in range(_2d_array_copy.shape[0]):

        shuffle_(_2d_array_copy[index])

    return _2d_array_copy


# ==============================================================================
# Matrix
# ==============================================================================


def function_on_2_2d_array(matrix_0, matrix_1, function):

    axis_0_size = matrix_0.shape[0]

    axis_1_size = matrix_1.shape[0]

    matrix = full((axis_0_size, axis_1_size), nan)

    for index_0 in range(axis_0_size):

        vector_0 = matrix_0[index_0]

        for index_1 in range(axis_1_size):

            vector_1 = matrix_1[index_1]

            matrix[index_0, index_1] = function(vector_0, vector_1)

    return matrix


# ==============================================================================
# ND array
# ==============================================================================


def check_is_not_na(nd_array):

    return logical_not(isna(nd_array))


# ==============================================================================
# Number array
# ==============================================================================


def clip(number_array, standard_deviation):

    mean = number_array.mean()

    margin = number_array.std() * standard_deviation

    return number_array.clip(min=mean - margin, max=mean + margin)


def shift_min(number_array, min):

    if min == "0<":

        min = number_array[0 < number_array].min()

    return min - number_array.min() + number_array


def log(number_array, log_base=2):

    return {2: log2, "e": loge, 10: log10}[log_base](number_array)


def normalize(number_array, method, rank_method="average"):

    if method == "-0-":

        return (number_array - number_array.mean()) / number_array.std()

    elif method == "0-1":

        min = number_array.min()

        return (number_array - min) / (number_array.max() - min)

    elif method == "sum":

        return number_array / number_array.sum()

    elif method == "rank":

        return rankdata(number_array, method=rank_method).reshape(number_array.shape)


def normalize_nd(array, axis, method, rank_method="average"):

    return apply_along_axis(normalize, axis, array, method, rank_method)


def guess_type(number_array, category_max_n=16):

    if all(isinstance(number, integer) for number in number_array.ravel()):

        category_n = unique(number_array).size

        if category_n <= 2:

            return "binary"

        elif category_n <= category_max_n:

            return "categorical"

    return "continuous"


# ==============================================================================
# Number array with NaN
# ==============================================================================


def check_is_not_nan(number_array):

    return logical_not(isnan(number_array))


def get_not_nan_unique(number_array):

    return unique(number_array[check_is_not_nan(number_array)])


def function_on_1_number_array_not_nan(
    number_array, function, *arg_, update=False, **kwarg_
):

    is_ = check_is_not_nan(number_array)

    returned = function(number_array[is_], *arg_, **kwarg_)

    if update:

        number_array_copy = full(number_array.shape, nan)

        number_array_copy[is_] = returned

        return number_array_copy

    return returned


def function_on_2_number_array_not_nan(
    number_array_0, number_array_1, function, *arg_, **kwarg_
):

    is_ = logical_and(
        check_is_not_nan(number_array_0), check_is_not_nan(number_array_1)
    )

    return function(number_array_0[is_], number_array_1[is_], *arg_, **kwarg_)
