from numpy import asarray, full
from pandas import isna, read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


# TODO: refactor
def get_gene_symbol(hgnc_column_to_values=None):

    if hgnc_column_to_values is None:

        hgnc_column_to_values = {"Locus type": ("gene with protein product",)}

    table = read_csv("{}/hgnc_gene_group.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t")

    is_selected = full(table.shape[0], True)

    for column, values in hgnc_column_to_values.items():

        is_selected &= asarray(
            tuple(not isna(value) and value in values for value in table.loc[:, column].to_numpy())
        )

        print(
            "Selected {}/{} gene by {}.".format(
                is_selected.sum(), is_selected.size, column
            )
        )

    return tuple(
        table.loc[is_selected, table.columns.str.contains("symbol")]
        .unstack()
        .dropna()
        .unique()
    )
