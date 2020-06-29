from numpy import asarray, full
from pandas import isna, read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def get_gene_symbol(hgnc_column_values=None):

    if hgnc_column_values is None:

        hgnc_column_values = {"Locus type": ("gene with protein product",)}

    hgnc = read_csv("{}/hgnc_gene_group.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t")

    selected = full(hgnc.shape[0], True)

    for column_name, values in hgnc_column_values.items():

        selected &= asarray(
            tuple(not isna(value) and value in values for value in hgnc[column_name])
        )

        print(
            "Selected {}/{} gene by {}.".format(
                selected.sum(), selected.size, column_name
            )
        )

    return tuple(
        hgnc.loc[selected, hgnc.columns.str.contains("symbol")]
        .unstack()
        .dropna()
        .unique()
    )
