from pandas import DataFrame

from .cluster_matrix import cluster_matrix
from .normalize_dataframe import normalize_dataframe
from .plot_heat_map import plot_heat_map


def plot_mf(ws, hs):

    for i, w in enumerate(ws):

        if not isinstance(w, DataFrame):

            w = DataFrame(w)

        plot_heat_map(
            normalize_dataframe(w.iloc[cluster_matrix(w.values, 0), :], 1, "-0-"),
            layout={"title": {"text": "W {}".format(i)}},
        )

    for i, h in enumerate(hs):

        if not isinstance(h, DataFrame):

            h = DataFrame(h)

        plot_heat_map(
            normalize_dataframe(h.iloc[:, cluster_matrix(h.values, 1)], 0, "-0-"),
            layout={"title": {"text": "H {}".format(i)}},
        )
