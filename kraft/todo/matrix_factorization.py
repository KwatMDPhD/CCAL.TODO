from numpy import apply_along_axis, asarray, dot, full, nan, sum
from numpy.linalg import norm, pinv
from numpy.random import random_sample, seed
from pandas import DataFrame
from scipy.optimize import nnls
from sklearn.decomposition import NMF

from .array import normalize
from .clustering import cluster
from .CONSTANT import RANDOM_SEED
from .plot import plot_heat_map, plot_plotly


def factorize_matrix_by_nmf(
    v, r, solver="cd", tolerance=1e-6, n_iteration=int(1e3), random_seed=RANDOM_SEED
):

    model = NMF(
        n_components=r,
        solver=solver,
        tol=tolerance,
        max_iter=n_iteration,
        random_state=random_seed,
    )

    return model.fit_transform(v), model.components_, model.reconstruction_err_


def factorize_matrix(
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


def plot_matrix_factorization(ws, hs, errors=None, axis_size=320, directory_path=None):

    axis_size_ = axis_size * 1.618

    for w_index, w in enumerate(ws):

        w = apply_along_axis(normalize, 1, w[cluster(w)[0], :], "-0-")

        layout_factor_axis = {"title": {"text": "Factor"}, "dtick": 1}

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/w{}.html".format(directory_path, w_index)

        plot_heat_map(
            w,
            layout={
                "height": axis_size_,
                "width": axis_size,
                "title": {"text": "W{}".format(w_index)},
                "xaxis": layout_factor_axis,
            },
            html_file_path=html_file_path,
        )

    for h_index, h in enumerate(hs):

        h = apply_along_axis(normalize, 0, h[:, cluster(h.T)[0]], "-0-")

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/h{}.html".format(directory_path, w_index)

        plot_heat_map(
            h,
            layout={
                "height": axis_size,
                "width": axis_size_,
                "title": {"text": "H{}".format(h_index)},
                "yaxis": layout_factor_axis,
            },
            html_file_path=html_file_path,
        )

    if errors is not None:

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = "{}/error.html".format(directory_path)

        plot_plotly(
            {
                "layout": {
                    "xaxis": {"title": "Iteration"},
                    "yaxis": {"title": "Error"},
                    "annotations": [
                        {
                            "x": error.size - 1,
                            "y": error[-1],
                            "text": "{:.2e}".format(error[-1]),
                        }
                        for error in errors
                    ],
                },
                "data": [
                    {"name": error_axis, "y": error}
                    for error_axis, error in enumerate(errors)
                ],
            },
            html_file_path=html_file_path,
        )


def solve_ax_b(a, b, method):

    a_ = a.values

    b_ = b.values

    if method == "pinv":

        x = dot(pinv(a_), b_)

    elif method == "nnls":

        x = full((a.shape[1], b.shape[1]), nan)

        for i in range(b.shape[1]):

            x[:, i] = nnls(a_, b_[:, i])[0]

    return DataFrame(x, index=a.columns, columns=b.columns)
