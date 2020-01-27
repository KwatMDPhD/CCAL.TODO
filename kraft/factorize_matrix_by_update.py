from numpy import full, nan
from numpy.linalg import norm
from numpy.random import random_sample, seed

from .RANDOM_SEED import RANDOM_SEED
from .update_matrix_factorization_h import update_matrix_factorization_h
from .update_matrix_factorization_w import update_matrix_factorization_w


def factorize_matrix_by_update(
    v, r, tolerance=1e-6, n_iteration=int(1e6), random_seed=RANDOM_SEED
):

    errors = full(n_iteration + 1, nan)

    seed(seed=random_seed)

    w = random_sample(size=(v.shape[0], r))

    h = random_sample(size=(r, v.shape[1]))

    errors[0] = norm(v - w @ h)

    for i in range(n_iteration):

        w = update_matrix_factorization_w(v, w, h)

        h = update_matrix_factorization_h(v, w, h)

        error = norm(v - w @ h)

        errors[i + 1] = error

        error_ = errors[i]

        if ((error_ - error) / error_) < tolerance:

            break

    return w, h, errors
