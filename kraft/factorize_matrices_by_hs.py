from numpy import asarray, full, nan, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed

from .RANDOM_SEED import RANDOM_SEED
from .update_h_by_multiplicative_update import update_h_by_multiplicative_update


def factorize_matrices_by_hs(
    vs, r, weights=None, tolerance=1e-6, n_iteration=int(1e6), random_seed=RANDOM_SEED
):

    assert len(set(v.shape[0] for v in vs)) == 1

    n_v = len(vs)

    errors = full((n_v, n_iteration + 1), nan)

    seed(seed=random_seed)

    w = random_sample(size=(vs[0].shape[0], r))

    hs = [random_sample(size=(r, v.shape[1])) for v in vs]

    errors[:, 0] = [norm(vs[i] - w @ hs[i]) for i in range(n_v)]

    v_0_norm = norm(vs[0])

    if weights is None:

        weights = [v_0_norm / norm(v) for v in vs]

    n_per_print = max(1, n_iteration // 100)

    for j in range(n_iteration):

        if j % n_per_print == 0:

            print("(r={}) {}/{}...".format(r, j + 1, n_iteration))

        top = sum([weights[i] * vs[i] @ hs[i].T for i in range(n_v)], axis=0)

        bottom = sum([weights[i] * w @ hs[i] @ hs[i].T for i in range(n_v)], axis=0)

        w *= top / bottom

        hs = [update_h_by_multiplicative_update(vs[i], w, hs[i]) for i in range(n_v)]

        j_1_errors = asarray([norm(vs[i] - w @ hs[i]) for i in range(n_v)])

        errors[:, j + 1] = j_1_errors

        j_errors = errors[:, j]

        if (((j_errors - j_1_errors) / j_errors) < tolerance).all():

            break

    return w, hs, errors
