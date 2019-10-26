from numpy import full, nan, sum
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from .RANDOM_SEED import RANDOM_SEED
from .update_w_by_multiplicative_update import update_w_by_multiplicative_update


def mf_vs_ws_h(vs, r, weights=None, n_iteration=int(1e3), random_seed=RANDOM_SEED):

    assert len(set(v.shape[1] for v in vs)) == 1

    n = len(vs)

    errors = full((n, n_iteration), nan)

    seed(seed=random_seed)

    ws = [random_sample(size=(v.shape[0], r)) for v in vs]

    h = random_sample(size=(r, vs[0].shape[1]))

    v_0_norm = compute_matrix_norm(vs[0])

    if weights is None:

        weights = [v_0_norm / compute_matrix_norm(v) for v in vs]

    for j in range(n_iteration):

        top = sum([weights[i] * ws[i].T @ vs[i] for i in range(n)], axis=0)

        bottom = sum([weights[i] * ws[i].T @ ws[i] @ h for i in range(n)], axis=0)

        h *= top / bottom

        ws = [update_w_by_multiplicative_update(vs[i], ws[i], h) for i in range(n)]

        errors[:, j] = [compute_matrix_norm(vs[i] - ws[i] @ h) for i in range(n)]

    return ws, h, errors
