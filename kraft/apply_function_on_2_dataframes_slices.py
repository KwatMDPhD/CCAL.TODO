from pandas import DataFrame

from .apply_function_on_2_2d_arrays_slices import apply_function_on_2_2d_arrays_slices
from .cluster_2d_array import cluster_2d_array
from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)
from .plot_heat_map import plot_heat_map


def apply_function_on_2_dataframes_slices(
    dataframe_0,
    dataframe_1,
    axis,
    function=compute_information_coefficient_between_2_1d_arrays,
    title=None,
    name_0=None,
    name_1=None,
    file_path_prefix=None,
):

    comparison = apply_function_on_2_2d_arrays_slices(
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
            cluster_2d_array(comparison.values, 0),
            cluster_2d_array(comparison.values, 1),
        ],
        title=title,
        xaxis_title=name_1,
        yaxis_title=name_0,
        html_file_path=html_file_path,
    )

    return comparison
