from pandas import DataFrame


def binarize(series):

    dataframe = DataFrame(index=series.unique(), columns=series.index)

    dataframe.index.name = series.name

    for str_ in dataframe.index:

        dataframe.loc[str_] = series == str_

    return dataframe.astype(int)
