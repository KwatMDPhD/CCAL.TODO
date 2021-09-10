from pandas import read_csv

from ..constant import data_directory


def _map_broad():

    return read_csv(
        "{}cell_line_name_rename.tsv.gz".format(data_directory),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()
