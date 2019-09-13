from os.path import join

from pandas import DataFrame, read_csv
from scipy.io import mmread

from .get_dataframe_fraction_good_on_axis import get_dataframe_fraction_good_on_axis
from .process_feature_x_sample import process_feature_x_sample
from .select_series_indices import select_series_indices


def read_process_write_gene_x_cell(
    mtx_file_path,
    index_file_path,
    column_file_path,
    output_directory_path,
    minimum_fraction_cell_with_gene_signal=None,
    minimum_fraction_gene_z_score=-1,
):

    gene_x_cell = DataFrame(
        mmread(mtx_file_path).toarray(),
        index=read_csv(index_file_path, sep="\t", header=None).iloc[:, 1],
        columns=read_csv(column_file_path, sep="\t", header=None, squeeze=True),
    )

    gene_x_cell.index.name = "Gene"

    if gene_x_cell.index.has_duplicates:

        print("Gene duplicated; merging duplicates with median...")

        gene_x_cell = gene_x_cell.groupby(level=0).median()

    gene_x_cell.columns.name = "Cell"

    assert not gene_x_cell.columns.has_duplicates

    gene_x_cell = gene_x_cell.sort_index().sort_index(axis=1)

    output_file_path = join(output_directory_path, "gene_x_cell.tsv")

    gene_x_cell.to_csv(output_file_path, sep="\t")

    gene_x_cell__clean__log = process_feature_x_sample(
        gene_x_cell,
        nanize=0,
        min_n_not_na_value=3,
        shift_as_necessary_to_achieve_min_before_logging=1,
        log_base=2,
        plot=False,
    )

    gene_x_cell__clean__log.to_csv(
        output_file_path.replace(".tsv", ".clean.log.tsv"), sep="\t"
    )

    gene_x_cell__clean__log.fillna(0).to_csv(
        output_file_path.replace(".tsv", ".clean.log.na0.tsv"), sep="\t"
    )

    if minimum_fraction_cell_with_gene_signal is None:

        genes = gene_x_cell__clean__log.index

    else:

        genes = select_series_indices(
            get_dataframe_fraction_good_on_axis(gene_x_cell__clean__log, 1),
            ">",
            thresholds=(minimum_fraction_cell_with_gene_signal,),
            layout={
                "title": {"text": "Genes"},
                "yaxis": {"title": {"text": "Fraction Cell"}},
            },
        )

    gene_x_cell__clean__log__selected_gene = gene_x_cell__clean__log.loc[genes]

    if minimum_fraction_gene_z_score is None:

        cells = gene_x_cell__clean__log__selected_gene.columns

    else:

        cells = select_series_indices(
            get_dataframe_fraction_good_on_axis(
                gene_x_cell__clean__log__selected_gene, 0
            ),
            ">",
            standard_deviation=minimum_fraction_gene_z_score,
            layout={
                "title": {"text": "Cells"},
                "yaxis": {"title": {"text": "Fraction Gene"}},
            },
        )

    gene_x_cell__clean__log__selected_gene[cells].to_csv(
        output_file_path.replace(".tsv", ".clean.log.selected.tsv"), sep="\t"
    )
