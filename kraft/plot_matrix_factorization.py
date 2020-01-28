from numpy import apply_along_axis
from pandas import DataFrame

from .get_clustering_index import get_clustering_index
from .normalize import normalize
from .plot_heat_map import plot_heat_map


def plot_matrix_factorization(ws, hs):

    axis_size_0 = 480

    axis_size_1 = axis_size_0 * 1.618

    for i, w in enumerate(ws):

        if not isinstance(w, DataFrame):

            w = DataFrame(w)

        w = w.iloc[get_clustering_index(w.values, 0), :]

        w = DataFrame(
            apply_along_axis(normalize, 1, w.values, "-0-"),
            index=w.index,
            columns=w.columns,
        )

        plot_heat_map(
            w,
            layout={
                "height": axis_size_1,
                "width": axis_size_0,
                "title": {"text": "W{}".format(i)},
            },
        )

    for i, h in enumerate(hs):

        if not isinstance(h, DataFrame):

            h = DataFrame(h)

        h = h.iloc[:, get_clustering_index(h.values, 1)]

        h = DataFrame(
            apply_along_axis(normalize, 0, h.values, "-0-"),
            index=h.index,
            columns=h.columns,
        )

        plot_heat_map(
            h,
            layout={
                "height": axis_size_0,
                "width": axis_size_1,
                "title": {"text": "H{}".format(i)},
            },
        )
