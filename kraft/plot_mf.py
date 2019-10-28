from pandas import DataFrame

from .cluster_matrix import cluster_matrix
from .normalize_dataframe import normalize_dataframe
from .plot_heat_map import plot_heat_map


def plot_mf(ws, hs, directory_path):

    axis_size_0 = 560

    axis_size_1 = axis_size_0 * 1.618

    for i, w in enumerate(ws):

        if not isinstance(w, DataFrame):

            w = DataFrame(w)

        plot_heat_map(
            normalize_dataframe(w.iloc[cluster_matrix(w.values, 0), :], 1, "-0-"),
            layout={
                "height": axis_size_1,
                "width": axis_size_0,
                "title": {"text": "W{}".format(i)},
            },
            html_file_path="{}/w{}.html".format(directory_path, i),
        )

    for i, h in enumerate(hs):

        if not isinstance(h, DataFrame):

            h = DataFrame(h)

        plot_heat_map(
            normalize_dataframe(h.iloc[:, cluster_matrix(h.values, 1)], 0, "-0-"),
            layout={
                "height": axis_size_0,
                "width": axis_size_1,
                "title": {"text": "H{}".format(i)},
            },
            html_file_path="{}/h{}.html".format(directory_path, i),
        )
