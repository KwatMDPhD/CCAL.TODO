from os.path import join

from numpy import array, full
from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH


def _read(co_se):

    da = read_csv(
        join(DATA_DIRECTORY_PATH, "hgnc_complete_set.txt.gz"),
        sep="\t",
        low_memory=False,
    )

    se_ = full(da.shape[0], True)

    for co, se in co_se.items():

        print("Selecting by {}: {}".format(co, se))

        se_ &= array([an in se for an in da.loc[:, co].values])

        print("{}/{}".format(se_.sum(), se_.size))

    return da.loc[se_, :]
