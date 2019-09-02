from numpy import full, nan
from pandas import DataFrame, concat

from .call_function_with_multiprocess import call_function_with_multiprocess
from .compute_vector_context import compute_vector_context
from .split_dataframe import split_dataframe


def _make_context_matrix(
    dataframe,
    skew_t_pdf_fit_parameter,
    n_grid,
    degree_of_freedom_for_tail_reduction,
    multiply_distance_from_reference_argmax,
    global_location,
    global_scale,
    global_degree_of_freedom,
    global_shape,
):

    context_matrix = full(dataframe.shape, nan)

    n = dataframe.shape[0]

    n_per_print = max(1, n // 10)

    for i, (index, series) in enumerate(dataframe.iterrows()):

        if i % n_per_print == 0:

            print(f"({i + 1}/{n}) {index} ...")

        if skew_t_pdf_fit_parameter is None:

            n_data = location = scale = degree_of_freedom = shape = None

        else:

            n_data, location, scale, degree_of_freedom, shape = skew_t_pdf_fit_parameter.loc[
                index, ["N Data", "Location", "Scale", "Degree of Freedom", "Shape"]
            ]

        context_matrix[i] = compute_vector_context(
            series.values,
            n_data=n_data,
            location=location,
            scale=scale,
            degree_of_freedom=degree_of_freedom,
            shape=shape,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=degree_of_freedom_for_tail_reduction,
            multiply_distance_from_reference_argmax=multiply_distance_from_reference_argmax,
            global_location=global_location,
            global_scale=global_scale,
            global_degree_of_freedom=global_degree_of_freedom,
            global_shape=global_shape,
        )["context_like_array"]

    return DataFrame(context_matrix, index=dataframe.index, columns=dataframe.columns)


def make_context_matrix(
    dataframe,
    n_job=1,
    skew_t_pdf_fit_parameter=None,
    n_grid=1e3,
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
    output_file_path=None,
):

    context_matrix = concat(
        call_function_with_multiprocess(
            _make_context_matrix,
            (
                (
                    dataframe_,
                    skew_t_pdf_fit_parameter,
                    n_grid,
                    degree_of_freedom_for_tail_reduction,
                    multiply_distance_from_reference_argmax,
                    global_location,
                    global_scale,
                    global_degree_of_freedom,
                    global_shape,
                )
                for dataframe_ in split_dataframe(
                    dataframe, 0, min(dataframe.shape[0], n_job)
                )
            ),
            n_job,
        )
    )

    if output_file_path is not None:

        context_matrix.to_csv(output_file_path, sep="\t")

    return context_matrix
