from numpy import asarray, full, nan, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed

from .RANDOM_SEED import RANDOM_SEED


def factorize_matrices(
    vs,
    r,
    mode,
    weights=None,
    tolerance=1e-6,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
):

    n_v = len(vs)

    errors = full((n_v, n_iteration + 1), nan)

    def update_matrix_factorization_w(v, w, h):

        return w * (v @ h.T) / (w @ h @ h.T)

    def update_matrix_factorization_h(v, w, h):

        return h * (w.T @ v) / (w.T @ w @ h)

    seed(seed=random_seed)

    if mode == "ws":

        ws = [random_sample(size=(v.shape[0], r)) for v in vs]

        h = random_sample(size=(r, vs[0].shape[1]))

        errors[:, 0] = [norm(vs[i] - ws[i] @ h) for i in range(n_v)]

        v_0_norm = norm(vs[0])

        if weights is None:

            weights = [v_0_norm / norm(v) for v in vs]

        n_per_print = max(1, n_iteration // 10)

        for iteration_index in range(n_iteration):

            if iteration_index % n_per_print == 0:

                print("{}/{}...".format(r, iteration_index + 1, n_iteration))

            t = sum([weights[i] * ws[i].T @ vs[i] for i in range(n_v)], axis=0)

            b = sum([weights[i] * ws[i].T @ ws[i] @ h for i in range(n_v)], axis=0)

            h *= t / b

            ws = [update_matrix_factorization_w(vs[i], ws[i], h) for i in range(n_v)]

            e = asarray([norm(vs[i] - ws[i] @ h) for i in range(n_v)])

            errors[:, iteration_index + 1] = e

            e_ = errors[:, iteration_index]

            if (((e_ - e) / e_) < tolerance).all():

                break

        hs = (h,)

    elif mode == "hs":

        w = random_sample(size=(vs[0].shape[0], r))

        hs = [random_sample(size=(r, v.shape[1])) for v in vs]

        errors[:, 0] = [norm(vs[i] - w @ hs[i]) for i in range(n_v)]

        v_0_norm = norm(vs[0])

        if weights is None:

            weights = [v_0_norm / norm(v) for v in vs]

        n_per_print = max(1, n_iteration // 100)

        for iteration_index in range(n_iteration):

            if iteration_index % n_per_print == 0:

                print("{}/{}...".format(r, iteration_index + 1, n_iteration))

            t = sum([weights[i] * vs[i] @ hs[i].T for i in range(n_v)], axis=0)

            b = sum([weights[i] * w @ hs[i] @ hs[i].T for i in range(n_v)], axis=0)

            w *= t / b

            hs = [update_matrix_factorization_h(vs[i], w, hs[i]) for i in range(n_v)]

            e = asarray([norm(vs[i] - w @ hs[i]) for i in range(n_v)])

            errors[:, iteration_index + 1] = e

            e_ = errors[:, iteration_index]

            if (((e_ - e) / e_) < tolerance).all():

                break

        ws = (w,)

    return ws, hs, errors
