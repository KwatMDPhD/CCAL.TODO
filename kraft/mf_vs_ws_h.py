from numpy import asarray, full, nan, sum
from numpy.random import random_sample, seed

from .compute_matrix_norm import compute_matrix_norm
from .RANDOM_SEED import RANDOM_SEED
from .update_w_by_multiplicative_update import update_w_by_multiplicative_update


def mf_vs_ws_h(
    vs, r, weights=None, tolerance=1e-6, n_iteration=int(1e4), random_seed=RANDOM_SEED
):

    assert len(set(v.shape[1] for v in vs)) == 1

    n_v = len(vs)

    errors = full((n_v, n_iteration + 1), nan)

    seed(seed=random_seed)

    ws = [random_sample(size=(v.shape[0], r)) for v in vs]

    h = random_sample(size=(r, vs[0].shape[1]))

    errors[:, 0] = [compute_matrix_norm(vs[i] - ws[i] @ h) for i in range(n_v)]

    v_0_norm = compute_matrix_norm(vs[0])

    if weights is None:

        weights = [v_0_norm / compute_matrix_norm(v) for v in vs]

    n_per_print = max(1, n_iteration // 10)

    for j in range(n_iteration):

        if j % n_per_print == 0:

            print("(r={}) {}/{}...".format(r, j + 1, n_iteration))

        top = sum([weights[i] * ws[i].T @ vs[i] for i in range(n_v)], axis=0)

        bottom = sum([weights[i] * ws[i].T @ ws[i] @ h for i in range(n_v)], axis=0)

        h *= top / bottom

        ws = [update_w_by_multiplicative_update(vs[i], ws[i], h) for i in range(n_v)]

        j_1_errors = asarray(
            [compute_matrix_norm(vs[i] - ws[i] @ h) for i in range(n_v)]
        )

        errors[:, j + 1] = j_1_errors

        j_errors = errors[:, j]

        if (((j_errors - j_1_errors) / j_errors) < tolerance).all():

            break

    return ws, h, errors
