from pandas import read_csv


def read(pa):

    return read_csv(pa, skiprows=2, sep="\t", index_col=0).drop(
        labels=["Description"], axis=1
    )
