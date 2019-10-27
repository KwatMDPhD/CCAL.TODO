from os.path import join

from numpy import isnan
from pandas import DataFrame, Index

from .establish_path import establish_path
from .mf_vs_w_hs import mf_vs_w_hs
from .mf_vs_ws_h import mf_vs_ws_h
from .plot_errors import plot_errors
from .plot_mf import plot_mf


def mf_dataframes(
    dataframes, method, r, directory_path, plot_w_h=True, **mf_vs_keyword_arguments
):

    establish_path(directory_path, "directory")

    vs = tuple(dataframe.values for dataframe in dataframes)

    if method == "vs_ws_h":

        ws, h, errors = mf_vs_ws_h(vs, r, **mf_vs_keyword_arguments)

        hs = (h,)

    elif method == "vs_w_hs":

        w, hs, errors = mf_vs_w_hs(vs, r, **mf_vs_keyword_arguments)

        ws = (w,)

    index_factors = Index(("r{}_f{}".format(r, i) for i in range(r)), name="Factor")

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

    if plot_w_h:

        plot_mf(ws, hs, directory_path)

    plot_errors(
        tuple(errors_[~isnan(errors_)] for errors_ in errors),
        layout={"title": {"text": "MF method {} r {} ".format(method, r)}},
    )

    return ws, hs
