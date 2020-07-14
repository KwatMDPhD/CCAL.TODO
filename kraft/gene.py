from numpy import asarray, full
from pandas import isna, read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


# TODO: refactor
def get_gene_symbol(hgnc_column_to_values=None):

    if hgnc_column_to_values is None:

        hgnc_column_to_values = {"Locus type": ("gene with protein product",)}

    hgnc = read_csv("{}/hgnc_gene_group.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t")

    is_selected = full(hgnc.shape[0], True)

    for column, values in hgnc_column_to_values.items():

        is_selected &= asarray(
            tuple(not isna(value) and value in values for value in hgnc.loc[:, column])
        )

        print(
            "Selected {}/{} gene by {}.".format(
                is_selected.sum(), is_selected.size, column
            )
        )

    return tuple(
        hgnc.loc[is_selected, hgnc.columns.str.contains("symbol")]
        .unstack()
        .dropna()
        .unique()
    )
