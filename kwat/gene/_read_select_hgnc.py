from numpy import array, full
from pandas import read_csv

from ..constant import DATA_DIRECTORY


def _read_select_hgnc(co_se):

    da = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY),
        sep="\t",
        low_memory=False,
    )

    if co_se is not None:

        ge_ = da.loc[:, "symbol"].values

        se_ = full(ge_.size, True)

        for co, se in co_se.items():

            print("Selecting by {}: {}...".format(co, ", ".join(se)))

            se_ &= array(
                [isinstance(an, str) and an in se for an in da.loc[:, co].values]
            )

            print("{}/{}".format(se_.sum(), se_.size))

        da = da.loc[se_, :]

    return da
