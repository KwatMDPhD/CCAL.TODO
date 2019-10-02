from os.path import join

from pandas import DataFrame, Index

from .establish_path import establish_path
from .mf_vs_ws_h import mf_vs_ws_h
from .mf_vs_w_hs import mf_vs_w_hs
from .plot_errors import plot_errors
from .plot_mf import plot_mf


def mf_dataframes(dataframes, k, method, directory_path, plot_heat_map=True):

    establish_path(directory_path, "directory")

    if method == "vs_ws_h":

        vs = tuple(dataframe.values for dataframe in dataframes)

        ws, h, errors = mf_vs_ws_h(vs, k)

        hs = (h,)

    elif method == "vs_w_hs":

        w, hs, errors = mf_vs_w_hs(vs, k)

        ws = (w,)

    index_factors = Index(("Factor{}".format(i) for i in range(k)), name="Factor")

    ws = tuple(
        DataFrame(w, index=dataframe.index, columns=index_factors)
        for dataframe, w in zip(dataframes, ws)
    )

    hs = tuple(
        DataFrame(h, index=index_factors, columns=dataframe.columns)
        for dataframe, h in zip(dataframes, hs)
    )

    for i, w in enumerate(ws):

        w.to_csv(join(directory_path, "w{}.tsv".format(i)), sep="\t")

    for i, h in enumerate(hs):

        h.to_csv(join(directory_path, "h{}.tsv".format(i)), sep="\t")

    if plot_heat_map:

        plot_mf(ws, hs, directory_path)

    plot_errors(errors)

    return ws, hs
