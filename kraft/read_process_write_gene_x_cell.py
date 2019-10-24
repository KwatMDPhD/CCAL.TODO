from os.path import join

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

    gene_x_cell.to_csv(join(output_directory_path, "gene_x_cell.tsv"), sep="\t")
