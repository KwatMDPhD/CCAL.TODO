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


def factorize_with_nmf(
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


def _update_w(v, w, h):

    return w * (v @ h.T) / (w @ h @ h.T)


def _update_h(v, w, h):

    return h * (w.T @ v) / (w.T @ w @ h)


def _is_tolerable(errors, tolerance):

    e_, e = asarray(errors)[-2:]

    return ((e_ - e) / e_ < tolerance).all()


def factorize(
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

    return ws, hs, asarray(errors).T


def make_factor_label_(r):

    return asarray(tuple("Factor {}.{}".format(r, index) for index in range(r)))


def plot(
    w_,
    h_,
    axis_0_label__,
    axis_1_label__,
    axis_0_name_,
    axis_1_name_,
    error__,
    axis_factor_size=640,
    directory_path=None,
):

    axis_size = axis_factor_size * 1.618

    factor_axis = {"dtick": 1}

    for w_index, w in enumerate(w_):

        w = apply_along_axis(normalize, 1, w[cluster(w)[0], :], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}w_{}.html".format(directory_path, w_index)

        plot_heat_map(
            w,
            axis_0_label__[w_index],
            make_factor_label_(w.shape[1]),
            axis_0_name_[w_index],
            "Factor",
            layout={
                "height": axis_size,
                "width": axis_factor_size,
                "title": {"text": "W {}".format(w_index)},
                "xaxis": factor_axis,
            },
            file_path=file_path,
        )

    for h_index, h in enumerate(h_):

        h = apply_along_axis(normalize, 0, h[:, cluster(h.T)[0]], "-0-")

        if directory_path is None:

            file_path = None

        else:

            file_path = "{}h_{}.html".format(directory_path, w_index)

        plot_heat_map(
            h,
            make_factor_label_(h.shape[0]),
            axis_1_label__[h_index],
            "Factor",
            axis_1_name_[h_index],
            layout={
                "height": axis_factor_size,
                "width": axis_size,
                "title": {"text": "H {}".format(h_index)},
                "yaxis": factor_axis,
            },
            file_path=file_path,
        )

    if directory_path is None:

        file_path = None

    else:

        file_path = "{}error.html".format(directory_path)

    plot_plotly(
        {
            "data": [
                {"name": index, "y": error_} for index, error_ in enumerate(error__)
            ],
            "layout": {
                "xaxis": {"title": "Iteration"},
                "yaxis": {"title": "Error"},
                "annotations": [
                    {
                        "x": error_.size - 1,
                        "y": error_[-1],
                        "text": "{:.2e}".format(error_[-1]),
                    }
                    for error_ in error__
                ],
            },
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
