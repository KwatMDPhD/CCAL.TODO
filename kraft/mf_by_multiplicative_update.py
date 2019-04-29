from numpy import full, nan
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from .RANDOM_SEED import RANDOM_SEED
from .update_H_by_multiplicative_update import update_H_by_multiplicative_update
from .update_W_by_multiplicative_update import update_W_by_multiplicative_update


def mf_by_multiplicative_update(V, k, n_iteration=int(1e3), random_seed=RANDOM_SEED):

    R_norms = full(n_iteration + 1, nan)

    seed(seed=random_seed)

    W = random_sample(size=(V.shape[0], k))

    H = random_sample(size=(k, V.shape[1]))

    R_norms[0] = compute_matrix_norm(V - W @ H)

    for i in range(n_iteration):

        W = update_W_by_multiplicative_update(V, W, H)

        H = update_H_by_multiplicative_update(V, W, H)

        R_norms[i + 1] = compute_matrix_norm(V - W @ H)

    return W, H, R_norms
