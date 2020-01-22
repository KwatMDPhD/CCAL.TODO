from numpy import asarray, full, isnan, nan, sum
from numpy.linalg import norm
from numpy.random import random_sample, seed
from pandas import DataFrame, Index

from .plot_errors import plot_errors
from .plot_matrix_factorization import plot_matrix_factorization
from .RANDOM_SEED import RANDOM_SEED
from .update_matrix_factorization_h import update_matrix_factorization_h
from .update_matrix_factorization_w import update_matrix_factorization_w


def factorize_matrices(
    vs,
    r,
    mode,
    weights=None,
    tolerance=1e-6,
    n_iteration=int(1e6),
    random_seed=RANDOM_SEED,
    directory_path=None,
    plot=True,
):

    if mode == "ws":

        assert len(set(v.shape[1] for v in vs)) == 1

        n_v = len(vs)

        errors = full((n_v, n_iteration + 1), nan)

        seed(seed=random_seed)

        ws = [random_sample(size=(v.shape[0], r)) for v in vs]

        h = random_sample(size=(r, vs[0].shape[1]))

        errors[:, 0] = [norm(vs[i] - ws[i] @ h) for i in range(n_v)]

        v_0_norm = norm(vs[0])

        if weights is None:

            weights = [v_0_norm / norm(v) for v in vs]

        n_per_print = max(1, n_iteration // 10)

        for j in range(n_iteration):

            if j % n_per_print == 0:

                print("(r={}) {}/{}...".format(r, j + 1, n_iteration))

            top = sum([weights[i] * ws[i].T @ vs[i] for i in range(n_v)], axis=0)

            bottom = sum([weights[i] * ws[i].T @ ws[i] @ h for i in range(n_v)], axis=0)

            h *= top / bottom

            ws = [update_matrix_factorization_w(vs[i], ws[i], h) for i in range(n_v)]

            j_1_errors = asarray([norm(vs[i] - ws[i] @ h) for i in range(n_v)])

            errors[:, j + 1] = j_1_errors

            j_errors = errors[:, j]

            if (((j_errors - j_1_errors) / j_errors) < tolerance).all():

                break

        hs = (h,)

    elif mode == "hs":

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

            hs = [update_matrix_factorization_h(vs[i], w, hs[i]) for i in range(n_v)]

            j_1_errors = asarray([norm(vs[i] - w @ hs[i]) for i in range(n_v)])

            errors[:, j + 1] = j_1_errors

            j_errors = errors[:, j]

            if (((j_errors - j_1_errors) / j_errors) < tolerance).all():

                break

        ws = (w,)

    index_factors = Index(("r{}_f{}".format(r, i) for i in range(r)), name="Factor")

    ws = tuple(
        DataFrame(w, index=dataframe.index, columns=index_factors)
        for dataframe, w in zip(vs, ws)
    )

    hs = tuple(
        DataFrame(h, index=index_factors, columns=dataframe.columns)
        for dataframe, h in zip(vs, hs)
    )

    for i, w in enumerate(ws):

        w.to_csv("{}/w{}.tsv".format(directory_path, i), sep="\t")

    for i, h in enumerate(hs):

        h.to_csv("{}/h{}.tsv".format(directory_path, i), sep="\t")

    if plot:

        plot_matrix_factorization(ws, hs, directory_path)

    plot_errors(
        tuple(errors_[~isnan(errors_)] for errors_ in errors),
        layout={
            "title": {"text": "Matrix Factorization (mode {} r {})".format(mode, r)}
        },
    )

    return ws, hs
