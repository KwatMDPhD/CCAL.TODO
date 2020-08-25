from numpy import (
    asarray,
    integer,
    diff,
    full,
    isnan,
    log as loge,
    log2,
    log10,
    nan,
    nanmin,
    quantile,
    sort,
    unique,
)
from numpy.random import seed, shuffle as shuffle_
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED


def check_has_duplicate(number_array):

    return (1 < unique(number_array, return_counts=True)[1]).any()


def get_not_nan_unique(number_array):

    return unique(number_array[~isnan(number_array)])


def error_nan(number_array):

    assert not isnan(number_array).any()


def guess_type(number_array, n_max_category=16):

    error_nan(number_array)

    if all(isinstance(number, integer) for number in number_array.ravel()):

        n_unique = unique(number_array).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= n_max_category:

            return "categorical"

    return "continuous"


def clip(number_array, standard_deviation):

    error_nan(number_array)

    assert 0 <= standard_deviation

    mean = number_array.mean()

    margin = number_array.std() * standard_deviation

    return number_array.clip(min=mean - margin, max=mean + margin)


def shift_minimum(number_array, minimum):

    error_nan(number_array)

    if minimum == "0<":

        minimum = number_array[0 < number_array].min()

    return minimum - number_array.min() + number_array


def log(number_array, log_base="2"):

    error_nan(number_array)

    assert (0 < number_array).all()

    return {"2": log2, "e": loge, "10": log10}[log_base](number_array)


def normalize(number_array, method, rank_method="average"):

    error_nan(number_array)

    if method == "-0-":

        standard_deviation = number_array.std()

        assert standard_deviation != 0

        return (number_array - number_array.mean()) / standard_deviation

    elif method == "0-1":

        minimum = number_array.min()

        range = number_array.max() - minimum

        assert range != 0

        return (number_array - minimum) / range

    elif method == "sum":

        assert (0 <= number_array).all()

        sum = number_array.sum()

        assert sum != 0

        return number_array / sum

    elif method == "rank":

        return rankdata(number_array, method=rank_method).reshape(number_array.shape)


def ignore_nan_and_function_1(
    numbers, function, *function_arguments, update=False, **function_keyword_arguments
):

    is_good = ~isnan(numbers)

    return_ = function(
        numbers[is_good], *function_arguments, **function_keyword_arguments
    )

    if update:

        numbers_ = full(numbers.shape, nan)

        numbers_[is_good] = return_

        return numbers_

    else:

        return return_


def ignore_nan_and_function_2(
    numbers_0, numbers_1, function, *function_arguments, **function_keyword_arguments,
):

    is_good = ~isnan(numbers_0) & ~isnan(numbers_1)

    return function(
        numbers_0[is_good],
        numbers_1[is_good],
        *function_arguments,
        **function_keyword_arguments,
    )


def shuffle(matrix, axis, random_seed=RANDOM_SEED):

    error_nan(matrix)

    assert matrix.ndim == 2

    matrix = matrix.copy()

    seed(seed=random_seed)

    if axis == 0:

        for i in range(matrix.shape[1]):

            shuffle_(matrix[:, i])

    elif axis == 1:

        for i in range(matrix.shape[0]):

            shuffle_(matrix[i, :])

    return matrix


def check_is_sorted(vector):

    error_nan(vector)

    assert vector.ndim == 1

    differences = diff(vector)

    return (differences <= 0).all() or (0 <= differences).all()


def map_int(objects):

    object_to_i = {}

    i_to_object = {}

    i = 0

    for object_ in objects:

        if object_ not in object_to_i:

            object_to_i[object_] = i

            i_to_object[i] = object_

            i += 1

    return object_to_i, i_to_object


def apply_function_on_slices_from_2_matrices(matrix_0, matrix_1, axis, function):

    error_nan(matrix_0)

    error_nan(matrix_1)

    if axis == 1:

        matrix_0 = matrix_0.T

        matrix_1 = matrix_1.T

    matrix = full((matrix_0.shape[0], matrix_1.shape[0]), nan)

    for i_0 in range(matrix_0.shape[0]):

        array_0 = matrix_0[i_0]

        for i_1 in range(matrix_1.shape[0]):

            matrix[i_0, i_1] = function(array_0, matrix_1[i_1])

    return matrix


def check_is_in(array_1d, value_):

    value_ = {value: None for value in value_}

    return asarray(tuple(value in value_ for value in array_1d))


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

        is_extreme_ = (vector <= low) | (high <= vector)

    elif direction == "<":

        is_extreme_ = vector <= low

    elif direction == ">":

        is_extreme_ = high <= vector

    return is_extreme_
