from numpy import array, full
from pandas import read_csv

from ..constant import DATA_DIRECTORY


def _read_hgnc(co_se):

    da = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY),
        sep="\t",
        low_memory=False,
    )

    if co_se is not None:

        ge_ = da.loc[:, "symbol"].values

        bo_ = full(ge_.size, True)

        for co, se in co_se.items():

            print("Selecting by {}: {}...".format(co, ", ".join(se)))

            bo_ &= array(
                [isinstance(an, str) and an in se for an in da.loc[:, co].values]
            )

            print("{}/{}".format(bo_.sum(), bo_.size))

        da = da.loc[bo_, :]

    return da
