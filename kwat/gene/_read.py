from numpy import array, full
from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH


def _read(co_se):

    da = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        low_memory=False,
    )

    if co_se is None:

        return da

    else:

        ge_ = da.loc[:, "symbol"].values

        se_ = full(ge_.size, True)

        for co, se in co_se.items():

            print("Selecting by {}: {}".format(co, se))

            se_ &= array([an in se for an in da.loc[:, co].values])

            print("{}/{}".format(se_.sum(), se_.size))

        return da.loc[se_, :]
