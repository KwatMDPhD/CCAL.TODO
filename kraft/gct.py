from pandas import read_csv


def read(file_path):

    return read_csv(file_path, skiprows=2, sep="\t", index_col=0).drop(
        labels=["Description"], axis=1
    )
