from pandas import DataFrame, read_csv
from scipy.io import mmread

from .dataframe import error_axes
from .feature_x_sample import collapse, process


def read_process_write_gene_x_cell(
    mtx_file_path, index_file_path, column_file_path, directory_path
):

    gene_x_cell = DataFrame(
        data=mmread(mtx_file_path).toarray(),
        index=read_csv(index_file_path, sep="\t", header=None).iloc[:, 1],
        columns=read_csv(column_file_path, sep="\t", header=None, squeeze=True),
    )

    error_axes(gene_x_cell)

    gene_x_cell.index.name = "Gene"

    gene_x_cell = collapse(gene_x_cell)

    gene_x_cell.columns.name = "Cell"

    file_path = "{}gene_x_cell.tsv".format(directory_path)

    gene_x_cell.to_csv(file_path, sep="\t")

    gene_x_cell_log = process(gene_x_cell, log_shift=1, log_base=2)

    gene_x_cell_log.to_csv(file_path.replace(".tsv", "_log2.tsv"), sep="\t")
