from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH
from ..dictionary import clean, rename as dictionary_rename


def rename(na_, **ke_va):

    return dictionary_rename(
        [na.lower() for na in na_],
        clean(
            read_csv(
                "{}cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
                sep="\t",
                index_col=0,
                squeeze=True,
            ).to_dict()
        ),
        **ke_va
    )
