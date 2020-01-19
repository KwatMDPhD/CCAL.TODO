from numpy import full, nan
from numpy.linalg import norm
from numpy.random import random_sample, seed

from .RANDOM_SEED import RANDOM_SEED
from .update_h_by_multiplicative_update import update_h_by_multiplicative_update
from .update_w_by_multiplicative_update import update_w_by_multiplicative_update


def factorize_matrix_by_update(
    V, k, n_iteration=int(1e3), random_seed=RANDOM_SEED
):

    R_norms = full(n_iteration + 1, nan)

    seed(seed=random_seed)

    W = random_sample(size=(V.shape[0], k))

    H = random_sample(size=(k, V.shape[1]))

    R_norms[0] = norm(V - W @ H)

    for i in range(n_iteration):

        W = update_w_by_multiplicative_update(V, W, H)

        H = update_h_by_multiplicative_update(V, W, H)

        R_norms[i + 1] = norm(V - W @ H)

    return W, H, R_norms
