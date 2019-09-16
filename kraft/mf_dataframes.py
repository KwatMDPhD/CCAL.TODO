from os.path import join

from pandas import DataFrame, Index

from .mf_vs_ws_h import mf_vs_ws_h
from .plot_errors import plot_errors
from .plot_mf import plot_mf


def mf_dataframes(dataframes, k, directory_path):

    ws, h, errors = mf_vs_ws_h(tuple(dataframe.values for dataframe in dataframes), k)

    hs = (h,)

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

        w.to_csv(join(directory_path, "{}_w.tsv".format(i)), sep="\t")

    for i, h in enumerate(hs):

        h.to_csv(join(directory_path, "{}_h.tsv".format(i)), sep="\t")

    plot_mf(ws, hs)

    plot_errors(errors)

    return ws, hs
