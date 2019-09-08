from numpy.random import seed, shuffle

from .check_array_for_bad import check_array_for_bad
from .RANDOM_SEED import RANDOM_SEED


def shuffle_each_matrix_slice(
    matrix, axis, random_seed=RANDOM_SEED, raise_for_bad=True
):

    check_array_for_bad(matrix, raise_for_bad=raise_for_bad)

    matrix_ = matrix.copy()

    seed(seed=random_seed)

    if axis == 0:

        for i in range(matrix.shape[1]):

            shuffle(matrix_[:, i])

    elif axis == 1:

        for i in range(matrix.shape[0]):

            shuffle(matrix_[i, :])

    return matrix_
