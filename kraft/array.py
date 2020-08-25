from numpy import (
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
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED

# ==============================================================================


def check_is_in(array_1d, lookup_array_1d):

    lookup = {value: None for value in lookup_array_1d}

    return asarray(tuple(value in lookup for value in array_1d))


def map_integer(array_1d):

    value_to_integer = {}

    integer_to_value = {}

    integer = 0

    for value in array_1d:

        value_to_integer[value] = integer

        integer_to_value[integer] = value

        integer += 1

    return value_to_integer, integer_to_value


# ==============================================================================


def check_is_all_sorted(vector):

    difference_ = diff(vector)

    return (difference_ <= 0).all() or (0 <= difference_).all()


def check_is_extreme(vector, direction, low_and_high=(), n=0, standard_deviation=0):

    if len(low_and_high) == 2:

        low, high = low_and_high

    elif 0 < n:

        if n < 1:

            low = quantile(vector, n)

            high = quantile(vector, 1 - n)

        else:

            vector_sorted = sort(vector)

            low = vector_sorted[n - 1]

            high = vector_sorted[-n]

    elif 0 < standard_deviation:

        mean = vector.mean()

        margin = vector.std() * standard_deviation

        low = mean - margin

        high = mean + margin

    if direction == "<>":

        is_ = logical_or(vector <= low, high <= vector)

    elif direction == "<":

        is_ = vector <= low

    elif direction == ">":

        is_ = high <= vector

    return is_


# ==============================================================================


def shuffle(array_2d, random_seed=RANDOM_SEED):

    array_2d_copy = array_2d.copy()

    seed(seed=random_seed)

    for axis_0_index in range(array_2d_copy.shape[0]):

        shuffle_(array_2d_copy[axis_0_index])

    return array_2d_copy


# ==============================================================================


def function_on_2_array_2d(matrix_0, matrix_1, function):

    axis_0_size = matrix_0.shape[0]

    axis_1_size = matrix_1.shape[0]

    matrix = full((axis_0_size, axis_1_size), nan)

    for index_0 in range(axis_0_size):

        array_1d_0 = matrix_0[index_0]

        for index_1 in range(axis_1_size):

            array_1d_1 = matrix_1[index_1]

            matrix[index_0, index_1] = function(array_1d_0, array_1d_1)

    return matrix


# ==============================================================================


def clip(number_array, standard_deviation):

    mean = number_array.mean()

    margin = number_array.std() * standard_deviation

    return number_array.clip(min=mean - margin, max=mean + margin)


def shift_minimum(number_array, minimum):

    if minimum == "0<":

        minimum = number_array[0 < number_array].min()

    return minimum - number_array.min() + number_array


def log(number_array, log_base="e"):

    return {"2": log2, "e": loge, "10": log10}[log_base](number_array)


def normalize(number_array, method, rank_method="average"):

    if method == "-0-":

        return (number_array - number_array.mean()) / number_array.std()

    elif method == "0-1":

        minimum = number_array.min()

        return (number_array - minimum) / (number_array.max() - minimum)

    elif method == "sum":

        return number_array / number_array.sum()

    elif method == "rank":

        return rankdata(number_array, method=rank_method).reshape(number_array.shape)


def guess_type(number_array, maximum_n_category=16):

    if all(isinstance(number, integer) for number in number_array.ravel()):

        n_unique = unique(number_array).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= maximum_n_category:

            return "categorical"

    return "continuous"


# ==============================================================================


def check_is_not_nan(number_array):

    return logical_not(isnan(number_array))


def get_not_nan_unique(number_array):

    return unique(number_array[check_is_not_nan(number_array)])


def function_on_1_number_array_not_nan(
    number_array, function, *args, update=False, **kwargs
):

    is_ = check_is_not_nan(number_array)

    returned = function(number_array[is_], *args, **kwargs)

    if update:

        number_array_copy = full(number_array.shape, nan)

        number_array_copy[is_] = returned

        return number_array_copy

    else:

        return returned


def function_on_2_number_array_not_nan(
    number_array_0, number_array_1, function, *args, **kwargs
):

    is_ = logical_and(
        check_is_not_nan(number_array_0), check_is_not_nan(number_array_1)
    )

    return function(number_array_0[is_], number_array_1[is_], *args, **kwargs)
