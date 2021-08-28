from pandas import read_csv

from ..constant import DATA_DIRECTORY
from ..dataframe import map_to


def _map_ensmus_to_gene():

    return map_to(
        read_csv("{}ensmus.tsv.gz".format(DATA_DIRECTORY), sep="\t"), "Gene name"
    )
