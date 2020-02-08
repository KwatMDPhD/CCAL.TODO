from numpy import asarray, sum
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

    def update_matrix_factorization_w(v, w, h):

        return w * (v @ h.T) / (w @ h @ h.T)

    def update_matrix_factorization_h(v, w, h):

        return h * (w.T @ v) / (w.T @ w @ h)

    def is_tolerable(errors, tolerance):

        e_, e = asarray(errors)[-2:]

        return ((e_ - e) / e_ < tolerance).all()

    seed(seed=random_seed)

    v_0_norm = norm(vs[0])

    if weights is None:

        weights = tuple(v_0_norm / norm(v) for v in vs)

    n_per_print = max(1, n_iteration // 10)

    if mode == "ws":

        ws = tuple(random_sample(size=(v.shape[0], r)) for v in vs)

        h = random_sample(size=(r, vs[0].shape[1]))

        errors = [tuple(norm(vs[i] - ws[i] @ h) for i in range(n_v))]

        for iteration_index in range(n_iteration):

            if iteration_index % n_per_print == 0:

                print("{}/{}...".format(iteration_index + 1, n_iteration))

            t = sum(tuple(weights[i] * ws[i].T @ vs[i] for i in range(n_v)), axis=0)

            b = sum(tuple(weights[i] * ws[i].T @ ws[i] @ h for i in range(n_v)), axis=0)

            h *= t / b

            ws = tuple(
                update_matrix_factorization_w(vs[i], ws[i], h) for i in range(n_v)
            )

            errors.append(tuple(norm(vs[i] - ws[i] @ h) for i in range(n_v)))

            if is_tolerable(errors, tolerance):

                break

        hs = (h,)

    elif mode == "hs":

        w = random_sample(size=(vs[0].shape[0], r))

        hs = tuple(random_sample(size=(r, v.shape[1])) for v in vs)

        errors = [tuple(norm(vs[i] - w @ hs[i]) for i in range(n_v))]

        for iteration_index in range(n_iteration):

            if iteration_index % n_per_print == 0:

                print("{}/{}...".format(iteration_index + 1, n_iteration))

            t = sum(tuple(weights[i] * vs[i] @ hs[i].T for i in range(n_v)), axis=0)

            b = sum(tuple(weights[i] * w @ hs[i] @ hs[i].T for i in range(n_v)), axis=0)

            w *= t / b

            hs = tuple(
                update_matrix_factorization_h(vs[i], w, hs[i]) for i in range(n_v)
            )

            errors.append(tuple(norm(vs[i] - w @ hs[i]) for i in range(n_v)))

            if is_tolerable(errors, tolerance):

                break

        ws = (w,)

    return ws, hs, asarray(errors).T
