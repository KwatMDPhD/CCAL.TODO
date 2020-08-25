from numpy import (
    asarray,
    logical_or,
    logical_not,
    logical_and,
    integer,
    diff,
    full,
    isnan,
    log as loge,
    log2,
    log10,
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


def shuffle(array_2d, random_seed=RANDOM_SEED):

    array_2d_copy = array_2d.copy()

    seed(seed=random_seed)

    for axis_0_index in range(array_2d_copy.shape[0]):

        shuffle_(array_2d_copy[axis_0_index])

    return array_2d_copy


def function_on_2_array_2d(array_2d_0, array_2d_1, function):

    axis_0_size = array_2d_0.shape[0]

    axis_1_size = array_2d_1.shape[0]

    array_2d = full((axis_0_size, axis_1_size), nan)

    for index_0 in range(axis_0_size):

        array_1d_0 = array_2d_0[index_0]

        for index_1 in range(axis_1_size):

            array_1d_1 = array_2d_1[index_1]

            array_2d[index_0, index_1] = function(array_1d_0, array_1d_1)

    return array_2d


# ==============================================================================


def check_is_not_nan(number_array):

    return logical_not(isnan(number_array))


def get_not_nan_unique(number_array):

    return unique(number_array[check_is_not_nan(number_array)])


def function_on_1_number_array_not_nan(
    number_array, function, *arguments, update=False, **keyword_arguments
):

    is_ = check_is_not_nan(number_array)

    returned = function(number_array[is_], *arguments, **keyword_arguments)

    if update:

        number_array_copy = full(number_array.shape, nan)

        number_array_copy[is_] = returned

        return number_array_copy

    else:

        return returned


def function_on_2_number_array_not_nan(
    number_array_0, number_array_1, function, *arguments, **keyword_arguments
):

    is_ = logical_and(
        check_is_not_nan(number_array_0), check_is_not_nan(number_array_1)
    )

    return function(
        number_array_0[is_], number_array_1[is_], *arguments, **keyword_arguments,
    )


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


def guess_type(number_array, n_max_category=16):

    if all(isinstance(number, integer) for number in number_array.ravel()):

        n_unique = unique(number_array).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= n_max_category:

            return "categorical"

    return "continuous"


def check_is_extreme(vector, direction, low_high=None, n=None, standard_deviation=None):

    if low_high is None:

        if n is not None:

            if n < 1:

                low = quantile(vector, n)

                high = quantile(vector, 1 - n)

            else:

                n = min(vector.size, n)

                vector_sorted = sort(vector)

                low = vector_sorted[n - 1]

                high = vector_sorted[-n]

        elif standard_deviation is not None:

            mean = vector.mean()

            margin = vector.std() * standard_deviation

            low = mean - margin

            high = mean + margin

    else:

        low, high = low_high

    if direction == "<>":

        is_ = logical_or(vector <= low, high <= vector)

    elif direction == "<":

        is_ = vector <= low

    elif direction == ">":

        is_ = high <= vector

    return is_


def check_is_all_sorted(vector):

    difference_ = diff(vector)

    return (difference_ <= 0).all() or (0 <= difference_).all()
