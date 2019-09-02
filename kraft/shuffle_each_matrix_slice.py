from numpy.random import seed, shuffle

from .check_array_for_bad import check_array_for_bad
from .RANDOM_SEED import RANDOM_SEED


def shuffle_each_matrix_slice(
    _matrix, axis, random_seed=RANDOM_SEED, raise_for_bad=True
):

    check_array_for_bad(_matrix, raise_for_bad=raise_for_bad)

    _matrix = _matrix.copy()

    seed(seed=random_seed)

    if axis == 0:

        for i in range(_matrix.shape[1]):

            shuffle(_matrix[:, i])

    elif axis == 1:

        for i in range(_matrix.shape[0]):

            shuffle(_matrix[i, :])

    return _matrix
