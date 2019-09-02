from pandas import DataFrame

from .apply_function_on_slices_from_2_matrices import (
    apply_function_on_slices_from_2_matrices,
)
from .cluster_matrix import cluster_matrix
from .compute_information_coefficient_between_2_vectors import (
    compute_information_coefficient_between_2_vectors,
)
from .plot_heat_map import plot_heat_map


def apply_function_on_slices_from_2_dataframes(
    dataframe_0,
    dataframe_1,
    axis,
    function=compute_information_coefficient_between_2_vectors,
    title=None,
    name_0=None,
    name_1=None,
    file_path_prefix=None,
):

    comparison = apply_function_on_slices_from_2_matrices(
        dataframe_0.values, dataframe_1.values, axis, function
    )

    if axis == 0:

        comparison = DataFrame(
            comparison, index=dataframe_0.index, columns=dataframe_1.index
        )

    elif axis == 1:

        comparison = DataFrame(
            comparison, index=dataframe_0.columns, columns=dataframe_1.columns
        )

    if file_path_prefix is None:

        html_file_path = None

    else:

        comparison.to_csv(f"{file_path_prefix}.tsv", sep="\t")

        html_file_path = f"{file_path_prefix}.html"

    plot_heat_map(
        comparison.iloc[
            cluster_matrix(comparison.values, 0), cluster_matrix(comparison.values, 1)
        ],
        title_text=title,
        xaxis_title_text=name_1,
        yaxis_title_text=name_0,
        html_file_path=html_file_path,
    )

    return comparison
