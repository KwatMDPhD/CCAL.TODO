from pandas import DataFrame, Series

from .normalize_array import normalize_array


def normalize_series_or_dataframe(
    series_or_dataframe, axis, method, rank_method="average"
):

    series_or_dataframe_normalized = type(series_or_dataframe)(
        normalize_array(
            series_or_dataframe.values,
            axis,
            method,
            rank_method=rank_method,
            raise_for_bad=False,
        )
    )

    if isinstance(series_or_dataframe, Series):

        series_or_dataframe_normalized.name = series_or_dataframe.name

        series_or_dataframe_normalized.index = series_or_dataframe.index

    elif isinstance(series_or_dataframe, DataFrame):

        series_or_dataframe_normalized.index = series_or_dataframe.index

        series_or_dataframe_normalized.columns = series_or_dataframe.columns

    return series_or_dataframe_normalized
