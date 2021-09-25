from os.path import join

from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH
from ..dictionary import clean, rename as dictionary_rename


def rename(na_, **ke_ar):

    return dictionary_rename(
        [na.lower() for na in na_],
        clean(
            read_csv(
                join(DATA_DIRECTORY_PATH, "cell_line_name_rename.tsv.gz"),
                sep="\t",
                index_col=0,
                squeeze=True,
            ).to_dict()
        ),
        **ke_ar,
    )
