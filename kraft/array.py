from numpy import diff, isnan, log as loge, log2, log10, nan, nanmin, unique
from numpy.random import seed, shuffle as shuffle_
from scipy.stats import rankdata

from .CONSTANT import RANDOM_SEED


def error_nan(numbers):

    assert not isnan(numbers).any()


def check_has_duplicate(array):

    return (unique(array, return_counts=True)[1] != 1).any()


def guess_type(numbers, n_max_category=16):

    error_nan(numbers)

    if all(float(number).is_integer() for number in numbers.flatten()):

        n_unique = unique(numbers).size

        if n_unique <= 2:

            return "binary"

        elif n_unique <= n_max_category:

            return "categorical"

    return "continuous"


def clip(numbers, standard_deviation):

    error_nan(numbers)

    assert 0 <= standard_deviation

    mean = numbers.mean()

    margin = numbers.std() * standard_deviation

    return numbers.clip(min=mean - margin, max=mean + margin)


def shift_minimum(numbers, minimum):

    error_nan(numbers)

    if minimum == "0<":

        minimum = numbers[0 < numbers].min()

    return numbers + minimum - nanmin(numbers)


def log(numbers, log_base="2"):

    error_nan(numbers)

    assert (0 < numbers).all()

    return {"2": log2, "e": loge, "10": log10}[log_base](numbers)


def normalize(numbers, method, rank_method="average"):

    error_nan(numbers)

    if method == "-0-":

        standard_deviation = numbers.std()

        assert standard_deviation != 0

        return (numbers - numbers.mean()) / standard_deviation

    elif method == "0-1":

        min_ = numbers.min()

        range_ = numbers.max() - min_

        assert range_ != 0

        return (numbers - min_) / range_

    elif method == "sum":

        assert (0 <= numbers).all()

        sum_ = numbers.sum()

        assert sum_ != 0

        return numbers / sum_

    elif method == "rank":

        return rankdata(numbers, method=rank_method).reshape(numbers.shape)


def ignore_nan_and_function_1(
    numbers, function, *function_arguments, update=False, **function_keyword_arguments
):

    is_good = ~isnan(numbers)

    if not is_good.any():

        return nan

    returned = function(
        numbers[is_good], *function_arguments, **function_keyword_arguments
    )

    if update:

        numbers = numbers.copy()

        numbers[is_good] = returned

        return numbers

    else:

        return returned


def ignore_nan_and_function_2(
    numbers_0, numbers_1, function, *function_arguments, **function_keyword_arguments,
):

    is_good = ~isnan(numbers_0) & ~isnan(numbers_1)

    if not is_good.any():

        return nan

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
