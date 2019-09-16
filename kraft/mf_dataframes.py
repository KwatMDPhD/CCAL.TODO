from pandas import DataFrame, Index

from .mf_vs_ws_h import mf_vs_ws_h
from .plot_errors import plot_errors
from .plot_mf import plot_mf


def mf_dataframes(dataframes, k, factor_name="Factor"):

    ws, h, rs = mf_vs_ws_h(tuple(dataframe.values for dataframe in dataframes), k)

    hs = (h,)

    index_factors = Index(("Factor{}".format(i) for i in range(k)), name=factor_name)

    w_dataframes = tuple(
        DataFrame(w, index=dataframe.index, columns=index_factors)
        for dataframe, w in zip(dataframes, ws)
    )

    h_dataframes = tuple(
        DataFrame(h, index=index_factors, columns=dataframe.columns)
        for dataframe, h in zip(dataframes, hs)
    )

    plot_mf(w_dataframes, h_dataframes)

    plot_errors(rs)

    return w_dataframes, h_dataframes
