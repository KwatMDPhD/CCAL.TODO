from os.path import join

from pandas import DataFrame

from .cluster_matrix import cluster_matrix
from .normalize_dataframe import normalize_dataframe
from .plot_heat_map import plot_heat_map


def plot_mf(ws, hs, directory_path=None):

    axis_size_0 = 500

    axis_size_1 = axis_size_0 * 1.618

    for i, w in enumerate(ws):

        if not isinstance(w, DataFrame):

            w = DataFrame(w)

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, "w{}.html".format(i))

        plot_heat_map(
            normalize_dataframe(w.iloc[cluster_matrix(w.values, 0), :], 1, "-0-"),
            layout={
                "height": axis_size_1,
                "width": axis_size_0,
                "title": {"text": "W{}".format(i)},
            },
            html_file_path=html_file_path,
        )

    for i, h in enumerate(hs):

        if not isinstance(h, DataFrame):

            h = DataFrame(h)

        if directory_path is None:

            html_file_path = None

        else:

            html_file_path = join(directory_path, "h{}.html".format(i))

        plot_heat_map(
            normalize_dataframe(h.iloc[:, cluster_matrix(h.values, 1)], 0, "-0-"),
            layout={
                "height": axis_size_0,
                "width": axis_size_1,
                "title": {"text": "H{}".format(i)},
            },
            html_file_path=html_file_path,
        )
