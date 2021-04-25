from pandas import read_csv


def read(path):

    return read_csv(path, skiprows=2, sep="\t", index_col=0).drop(
        labels=["Description"], axis=1
    )
