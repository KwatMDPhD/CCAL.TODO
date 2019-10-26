from os.path import join
from .process_feature_x_sample import process_feature_x_sample

from pandas import DataFrame, read_csv
from scipy.io import mmread


def read_process_write_gene_x_cell(
    mtx_file_path, index_file_path, column_file_path, output_directory_path
):

    gene_x_cell = DataFrame(
        mmread(mtx_file_path).toarray(),
        index=read_csv(index_file_path, sep="\t", header=None).iloc[:, 1],
        columns=read_csv(column_file_path, sep="\t", header=None, squeeze=True),
    )

    gene_x_cell.index.name = "Gene"

    if gene_x_cell.index.has_duplicates:

        print("Merging duplicated genes with median...")

        gene_x_cell = gene_x_cell.groupby(level=0).median()

    gene_x_cell.columns.name = "Cell"

    assert not gene_x_cell.columns.has_duplicates

    gene_x_cell = gene_x_cell.sort_index().sort_index(axis=1)

    tsv_file_path = join(output_directory_path, "gene_x_cell.tsv")

    gene_x_cell.to_csv(tsv_file_path, sep="\t")

    gene_x_cell_log = process_feature_x_sample(
        gene_x_cell, shift_as_necessary_to_achieve_min_before_logging=1, log_base=2
    )

    gene_x_cell_log.to_csv(tsv_file_path.replace(".tsv", "log2.tsv"), sep="\t")
