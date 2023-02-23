from pandas import read_csv


def read(pa):
    return read_csv(pa, sep="\t", comment="#", header=None, low_memory=False)
