from os.path import join

from .process_feature_x_sample import process_feature_x_sample
from .read_matrix_market import read_matrix_market


def read_gene_x_cell(
    mtx_file_path, genes_file_path, barcodes_file_path, output_directory_path
):

    gene_x_cell = read_matrix_market(
        mtx_file_path,
        genes_file_path,
        barcodes_file_path,
        index_name="Gene",
        column_name="Cell",
    )

    gene_x_cell = gene_x_cell.groupby(level=0).median()

    output_file_path = join(output_directory_path, "gene_x_cell.tsv")

    gene_x_cell.to_csv(output_file_path, sep="\t")

    gene_x_cell__clean__log = process_feature_x_sample(
        gene_x_cell,
        nanize=0,
        min_n_not_na_value=3,
        min_n_not_na_unique_value=1,
        shift_as_necessary_to_achieve_min_before_logging=1,
        log_base=2,
        plot=True,
    )

    return gene_x_cell__clean__log

    # gene_x_cell__clean__log.to_csv(
    #     output_file_path.replace(".tsv", ".clean.log.tsv"), sep="\t"
    # )

    # gene_x_cell__clean__log.fillna(0).to_csv(
    #     output_file_path.replace(".tsv", ".clean.log.na0.tsv"), sep="\t"
    # )
