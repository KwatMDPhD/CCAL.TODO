from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def _map_broad():

    return read_csv(
        "{}cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        "\t",
        index_col=0,
        squeeze=True,
    ).to_dict()


def rename(na_):

    na_ce = _map_broad()

    ce_ = []

    fa_ = []

    for na in na_:

        nalo = na.lower()

        if nalo in na_ce:

            ce_.append(na_ce[nalo])

        else:

            ce_.append(None)

            fa_.append(na)

    if 0 < len(fa_):

        print("Failed {}.".format(sorted(set(fa_))))

    return ce_
