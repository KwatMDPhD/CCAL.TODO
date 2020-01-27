from numpy import unique
from pandas import DataFrame


def binarize(series):

    series = series.astype(str)

    str_x_index = DataFrame(index=unique(series), columns=series.index)

    for str_ in str_x_index.index:

        str_x_index.loc[str_] = series == str_

    str_x_index.index.name = series.name

    return str_x_index.astype(int)
