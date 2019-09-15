from pandas import concat

from .call_function_with_multiprocess import call_function_with_multiprocess
from .make_context_matrix_ import make_context_matrix_
from .split_dataframe import split_dataframe


def make_context_matrix(
    dataframe,
    n_job=1,
    skew_t_pdf_fit_parameter=None,
    n_grid=int(1e3),
    degree_of_freedom_for_tail_reduction=1e8,
    multiply_distance_from_reference_argmax=False,
    global_location=None,
    global_scale=None,
    global_degree_of_freedom=None,
    global_shape=None,
    output_tsv_file_path=None,
):

    context_matrix = concat(
        call_function_with_multiprocess(
            make_context_matrix_,
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

    if output_tsv_file_path is not None:

        context_matrix.to_csv(output_tsv_file_path, sep="\t")

    return context_matrix
