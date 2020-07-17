from pandas import read_csv


def read(gct_file_path):

    return read_csv(gct_file_path, skiprows=2, sep="\t", index_col=0).drop(
        "Description", axis=1
    )
