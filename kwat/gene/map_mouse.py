from os.path import join

from numpy import unique
from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH


def _update(da, mo_hu, hu_mo):

    og_, sy_ = da.values.T

    hu_ = og_ == "human"

    mo_ = og_ == "mouse, laboratory"

    syh_ = unique(sy_[hu_])

    sym_ = unique(sy_[mo_])

    if syh_.size == 1:

        syh = syh_[0]

        for sym in sym_:

            mo_hu[sym] = syh

    if sym_.size == 1:

        sym = sym_[0]

        for syh in syh_:

            hu_mo[syh] = sym

    if 1 < syh_.size and 1 < sym_.size:

        print("-" * 80)

        print("\n".join("({}) {}".format(og, sy) for og, sy in zip(og_, sy_)))


def map_mouse():

    mo_hu = {}

    hu_mo = {}

    read_csv(
        join(DATA_DIRECTORY_PATH, "HOM_MouseHumanSequence.rpt.txt.gz"),
        sep="\t",
        usecols=[0, 1, 3],
        index_col=0,
    ).groupby(level=0).apply(_update, mo_hu, hu_mo)

    return mo_hu, hu_mo
