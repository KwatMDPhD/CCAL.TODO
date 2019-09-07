from .get_dataframe_slice_fraction_good import get_dataframe_slice_fraction_good
from .select_series_indices import select_series_indices


def select_gene_x_cell(
    gene_x_cell,
    minimum_fraction_cell_with_gene_signal=None,
    minimum_fraction_gene_z_score=-1,
):

    if minimum_fraction_cell_with_gene_signal is None:

        genes = gene_x_cell.index

    else:

        genes = select_series_indices(
            get_dataframe_slice_fraction_good(gene_x_cell, 1),
            ">",
            thresholds=(minimum_fraction_cell_with_gene_signal,),
            title={"text": "Genes"},
            xaxis={"title": {"text": "Ranking"}},
            yaxis={"title": {"text": "Fraction Cell"}},
        )

    print("Selected {} genes.".format(genes.size))

    if minimum_fraction_gene_z_score is None:

        cells = gene_x_cell.columns

    else:

        cells = select_series_indices(
            get_dataframe_slice_fraction_good(gene_x_cell.loc[genes], 0),
            ">",
            standard_deviation=minimum_fraction_gene_z_score,
            title={"text": "Cells"},
            xaxis={"title": {"text": "Ranking"}},
            yaxis={"title": {"text": "Fraction Gene"}},
        )

    print("Selected {} cells.".format(cells.size))

    return gene_x_cell.loc[genes, cells]
