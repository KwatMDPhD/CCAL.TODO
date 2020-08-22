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
from .table import untangle


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


def _update_matrix_factorization_w(v, w, h):

    return w * (v @ h.T) / (w @ h @ h.T)


def _update_matrix_factorization_h(v, w, h):

    return h * (w.T @ v) / (w.T @ w @ h)


def _is_tolerable(errors, tolerance):

    e_, e = asarray(errors)[-2:]

    return ((e_ - e) / e_ < tolerance).all()


def factorize_matrix(
    vs,
    mode,
    r,
    weights=None,
    tolerance=1e-6,
    n_iteration=int(1e3),
    random_seed=RANDOM_SEED,
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

            ws = tuple(
                _update_matrix_factorization_w(vs[i], ws[i], h) for i in range(n_v)
            )

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

            hs = tuple(
                _update_matrix_factorization_h(vs[i], w, hs[i]) for i in range(n_v)
            )

            errors.append(tuple(norm(vs[i] - w @ hs[i]) for i in range(n_v)))

            if _is_tolerable(errors, tolerance):

                break

        ws = (w,)

    return ws, hs, asarray(errors).T


def plot_matrix_factorization(
    ws, hs, errors, factor_axis_size=640, directory_path=None
):

    axis_size = factor_axis_size * 1.618

    layout_factor_axis = {"title": {"text": "Factor"}, "dtick": 1}

    for w_i, w in enumerate(ws):

        if isinstance(w, DataFrame):

            w, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(w)

        else:

            axis_0_labels = axis_1_labels = axis_0_name = axis_1_name = None

        w = apply_along_axis(normalize, 1, w[cluster(w)[0], :], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}w_{}.html".format(directory_path, w_i)

        plot_heat_map(
            w,
            axis_0_labels,
            axis_1_labels,
            axis_0_name,
            axis_1_name,
            layout={
                "height": axis_size,
                "width": factor_axis_size,
                "title": {"text": "W {}".format(w_i)},
                "xaxis": layout_factor_axis,
            },
            file_path=file_path,
        )

    for h_i, h in enumerate(hs):

        if isinstance(h, DataFrame):

            h, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(h)

        else:

            axis_0_labels = axis_1_labels = axis_0_name = axis_1_name = None

        h = apply_along_axis(normalize, 0, h[:, cluster(h.T)[0]], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}h_{}.html".format(directory_path, w_i)

        plot_heat_map(
            h,
            axis_0_labels,
            axis_1_labels,
            axis_0_name,
            axis_1_name,
            layout={
                "height": factor_axis_size,
                "width": axis_size,
                "title": {"text": "H {}".format(h_i)},
                "yaxis": layout_factor_axis,
            },
            file_path=file_path,
        )

    if errors is not None:

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}error.html".format(directory_path)

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
                "data": [{"name": i, "y": error} for i, error in enumerate(errors)],
            },
            file_path=file_path,
        )


def solve_ax_b(a, b, method):

    a_ = a.to_numpy()

    b_ = b.to_numpy()

    if method == "pinv":

        x = dot(pinv(a_), b_)

    elif method == "nnls":

        x = full((a.shape[1], b.shape[1]), nan)

        for i in range(b.shape[1]):

            x[:, i] = nnls(a_, b_[:, i])[0]

    return DataFrame(data=x, index=a.columns, columns=b.columns)
