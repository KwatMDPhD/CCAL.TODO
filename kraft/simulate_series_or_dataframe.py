from numpy import nan
from numpy.random import choice
from pandas import DataFrame, Series, Index

from .simulate_array import simulate_array


def simulate_series_or_dataframe(
    name_0, name_1, *simulate_array_arguments, break_dataframe=None
):

    array = simulate_array(*simulate_array_arguments)

    if len(array.shape) == 1:

        series_or_dataframe = Series(array, name=name_1)

    elif len(array.shape) == 2:

        series_or_dataframe = DataFrame(
            array,
            columns=Index(
                ("{}{}".format(name_1, i) for i in range(array.shape[1])), name="Column"
            ),
        )

    series_or_dataframe.index = Index(
        ("{}{}".format(name_0, i) for i in range(array.shape[0])), name="Index"
    )

    if len(series_or_dataframe.shape) == 2 and break_dataframe is not None:

        if break_dataframe < series_or_dataframe.shape[0]:

            for i in range(break_dataframe):

                series_or_dataframe.iloc[i] = i

        series_or_dataframe.loc[
            choice(
                series_or_dataframe.index,
                size=series_or_dataframe.shape[0] // break_dataframe,
                replace=False,
            ),
            choice(
                series_or_dataframe.columns,
                size=series_or_dataframe.shape[1] // break_dataframe,
                replace=False,
            ),
        ] = nan

    return series_or_dataframe
