from numpy import full, nan, sum
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from .RANDOM_SEED import RANDOM_SEED
from .update_h_by_multiplicative_update import update_h_by_multiplicative_update


def mf_with_vs_and_hs(
    Vs, k, weights=None, n_iteration=int(1e3), random_seed=RANDOM_SEED
):

    n = len(Vs)

    R_norms = full((n, n_iteration + 1), nan)

    seed(seed=random_seed)

    W = random_sample(size=(Vs[0].shape[0], k))

    Hs = [random_sample(size=(k, V.shape[1])) for V in Vs]

    R_norms[:, 0] = [compute_matrix_norm(Vs[i] - W @ Hs[i]) for i in range(n)]

    V_0_norm = compute_matrix_norm(Vs[0])

    if weights is None:

        weights = [V_0_norm / compute_matrix_norm(V) for V in Vs]

    for j in range(n_iteration):

        top = sum([weights[i] * Vs[i] @ Hs[i].T for i in range(n)], axis=0)

        bottom = sum([weights[i] * W @ Hs[i] @ Hs[i].T for i in range(n)], axis=0)

        W *= top / bottom

        Hs = [update_h_by_multiplicative_update(Vs[i], W, Hs[i]) for i in range(n)]

        R_norms[:, j + 1] = [compute_matrix_norm(Vs[i] - W @ Hs[i]) for i in range(n)]

    return W, Hs, R_norms
