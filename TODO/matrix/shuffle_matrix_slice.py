from numpy.random import seed, shuffle

from .is_array_bad import is_array_bad
from .RANDOM_SEED import RANDOM_SEED


def shuffle_matrix_slice(matrix, axis, random_seed=RANDOM_SEED, raise_if_bad=True):

    is_array_bad(matrix, raise_if_bad=raise_if_bad)

    matrix_ = matrix.copy()

    seed(seed=random_seed)

    if axis == 0:

        for i in range(matrix.shape[1]):

            shuffle(matrix_[:, i])

    elif axis == 1:

        for i in range(matrix.shape[0]):

            shuffle(matrix_[i, :])

    return matrix_
