from numpy import array, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed

from ..constant import random_seed


def factorize(
    vs,
    mode,
    r,
    weights=None,
    tolerance=1e-6,
    n_iteration=int(1e3),
    random_seed=random_seed,
):

    n_v = len(vs)

    seed(seed=random_seed)

    v_0_norm = norm(vs[0])

    if weights is None:

        weights = tuple(v_0_norm / norm(v) for v in vs)

    if mode == "ws":

        ws = tuple(random_sample(size=(v.shape[0], r)) for v in vs)

        h = random_sample(size=(r, vs[0].shape[1]))

        errors = [tuple(norm(vs[i] - ws[i] @ h) for i in range(n_v))]

        for _ in range(n_iteration):

            t = sum(tuple(weights[i] * ws[i].T @ vs[i] for i in range(n_v)), axis=0)

            b = sum(tuple(weights[i] * ws[i].T @ ws[i] @ h for i in range(n_v)), axis=0)

            h *= t / b

            ws = tuple(_update_w(vs[i], ws[i], h) for i in range(n_v))

            errors.append(tuple(norm(vs[i] - ws[i] @ h) for i in range(n_v)))

            if _is_tolerable(errors, tolerance):

                break

        hs = (h,)

    elif mode == "hs":

        w = random_sample(size=(vs[0].shape[0], r))

        hs = tuple(random_sample(size=(r, v.shape[1])) for v in vs)

        errors = [tuple(norm(vs[i] - w @ hs[i]) for i in range(n_v))]

        for _ in range(n_iteration):

            t = sum(tuple(weights[i] * vs[i] @ hs[i].T for i in range(n_v)), axis=0)

            b = sum(tuple(weights[i] * w @ hs[i] @ hs[i].T for i in range(n_v)), axis=0)

            w *= t / b

            hs = tuple(_update_h(vs[i], w, hs[i]) for i in range(n_v))

            errors.append(tuple(norm(vs[i] - w @ hs[i]) for i in range(n_v)))

            if _is_tolerable(errors, tolerance):

                break

        ws = (w,)

    return (ws, hs, array(errors).T)
