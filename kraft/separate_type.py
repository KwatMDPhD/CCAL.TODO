from numpy import nan
from pandas import DataFrame, concat

from .BAD_STR import BAD_STR
from .binarize import binarize
from .cast_builtin import cast_builtin
from .guess_type import guess_type


def separate_type(information_x_, bad_values=BAD_STR):

    continuous_dataframe_rows = []

    binary_dataframes = []

    for information, series in information_x_.iterrows():

        series = series.replace(bad_values, nan)

        if 1 < series.dropna().unique().size:

            try:

                is_continuous = guess_type(series.astype(float)) == "continuous"

            except ValueError:

                is_continuous = False

            if is_continuous:

                continuous_dataframe_rows.append(series.apply(cast_builtin))

            else:

                binary_x_ = binarize(series)

                binary_x_ = binary_x_.loc[~binary_x_.index.isna()]

                binary_x_.index = (
                    "({}) {}".format(binary_x_.index.name, str_)
                    for str_ in binary_x_.index
                )

                binary_dataframes.append(binary_x_)

    continuous_x_ = DataFrame(data=continuous_dataframe_rows)

    continuous_x_.index.name = information_x_.index.name

    binary_x_ = concat(binary_dataframes)

    binary_x_.index.name = information_x_.index.name

    return continuous_x_, binary_x_
