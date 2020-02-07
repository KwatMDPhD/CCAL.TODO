from numpy import unique
from pandas import DataFrame


def binarize(series):

    series = series.astype(str)

    dataframe = DataFrame(index=unique(series), columns=series.index)

    dataframe.index.name = series.name

    for str_ in dataframe.index:

        dataframe.loc[str_] = series == str_

    return dataframe.astype(int)
