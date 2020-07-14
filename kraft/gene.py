from numpy import asarray, full, unique
from pandas import isna, read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def get_gene_symbol(column_to_selections=None):

    if column_to_selections is None:

        column_to_selections = {"Locus group": ("protein-coding gene",)}

    table = read_csv("{}/hgnc_gene_group.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t")

    is_selected = full(table.shape[0], True)

    for label, selections in column_to_selections.items():

        print(label)

        is_selected &= asarray(
            tuple(
                not isna(object_) and object_ in selections
                for object_ in table.loc[:, label].to_numpy()
            )
        )

        print("{}/{}".format(is_selected.sum(), is_selected.size))

    genes = (
        table.loc[
            is_selected, ["symbol" in label for label in table.columns.to_numpy()],
        ]
        .to_numpy()
        .ravel()
    )

    return unique(genes[~isna(genes)])
