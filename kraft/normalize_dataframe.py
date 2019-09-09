from pandas import DataFrame

from .normalize_array import normalize_array
from .normalize_array_on_axis import normalize_array_on_axis


def normalize_dataframe(dataframe, axis, method, rank_method="average"):

    values = dataframe.values

    normalize_array_keyword_arguments = {
        "rank_method": rank_method,
        "raise_for_bad": False,
    }

    if axis is None:

        values = normalize_array(values, method, **normalize_array_keyword_arguments)

    else:

        values = normalize_array_on_axis(
            values, axis, method, **normalize_array_keyword_arguments
        )

    return DataFrame(values, index=dataframe.index, columns=dataframe.columns)
