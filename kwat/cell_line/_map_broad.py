from pandas import read_csv

from ..constant import DATA_DIRECTORY_PATH


def _map_broad():

    return read_csv(
        "{}cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()
