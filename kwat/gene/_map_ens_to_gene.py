from pandas import read_csv

from ..constant import data_directory
from ..dataframe import map_to


def _map_ens_to_gene():

    return map_to(
        read_csv("{}ens.tsv.gz".format(data_directory), sep="\t"), "Gene name"
    )
