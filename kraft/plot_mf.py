from os.path import join

from pandas import DataFrame

from .cluster_matrix import cluster_matrix
from .normalize_dataframe import normalize_dataframe
from .plot_heat_map import plot_heat_map


def plot_mf(ws, hs, directory_path):

    axis_size_0 = 640

    axis_size_1 = axis_size_0 * 1.618

    for i, w in enumerate(ws):

        if not isinstance(w, DataFrame):

            w = DataFrame(w)

        title_text = "W"

        html_file_path = join(directory_path, "w.html")

        if 1 < len(ws):

            title_text = "{}{}".format(title_text, i)

            html_file_path = html_file_path.replace(".html", "{}.html".format(i))

        plot_heat_map(
            normalize_dataframe(w.iloc[cluster_matrix(w.values, 0), :], 1, "-0-"),
            layout={
                "height": axis_size_1,
                "width": axis_size_0,
                "title": {"text": title_text},
            },
            html_file_path=html_file_path,
        )

    for i, h in enumerate(hs):

        if not isinstance(h, DataFrame):

            h = DataFrame(h)

        title_text = "H"

        html_file_path = join(directory_path, "h.html")

        if 1 < len(hs):

            title_text = "{}{}".format(title_text, i)

            html_file_path = html_file_path.replace(".html", "{}.html".format(i))

        plot_heat_map(
            normalize_dataframe(h.iloc[:, cluster_matrix(h.values, 1)], 0, "-0-"),
            layout={
                "height": axis_size_0,
                "width": axis_size_1,
                "title": {"text": title_text},
            },
            html_file_path=html_file_path,
        )
