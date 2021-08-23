from pandas import read_csv


def read(pa):

    return read_csv(pa, sep="\t", skiprows=2, index_col=0).drop(["Description"], axis=1)
