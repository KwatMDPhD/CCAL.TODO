from pandas import Series

from .normalize_array import normalize_array


def normalize_series(series, method, rank_method="average"):

    return Series(
        normalize_array(
            series.values, method, rank_method=rank_method, raise_for_bad=False
        ),
        name=series.name,
        index=series.index,
    )
