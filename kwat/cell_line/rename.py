from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH
from ..dictionary import clean


def rename(na_):

    na_re = clean(
        read_csv(
            "{}cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
            sep="\t",
            index_col=0,
            squeeze=True,
        ).to_dict()
    )

    re_ = []

    fa_ = []

    for na in na_:

        re = na_re.get(na.lower())

        re_.append(re)

        if re is None:

            fa_.append(na)

    if 0 < len(fa_):

        print("Failed {}.".format(sorted(set(fa_))))

    return re_
