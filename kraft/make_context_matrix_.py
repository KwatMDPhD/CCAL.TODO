from numpy import full, nan
from pandas import DataFrame

from .compute_vector_context import compute_vector_context


def make_context_matrix_(
    dataframe_,
    skew_t_pdf_fit_parameter,
    n_grid,
    degree_of_freedom_for_tail_reduction,
    multiply_distance_from_reference_argmax,
    global_location,
    global_scale,
    global_degree_of_freedom,
    global_shape,
):

    context_matrix = full(dataframe_.shape, nan)

    n = dataframe_.shape[0]

    n_per_print = max(1, n // 10)

    for i, (index, series) in enumerate(dataframe_.iterrows()):

        if i % n_per_print == 0:

            print("({}/{}) {}...".format(i + 1, n, index))

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

    return DataFrame(context_matrix, index=dataframe_.index, columns=dataframe_.columns)
