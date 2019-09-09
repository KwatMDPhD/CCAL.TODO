from numpy import apply_along_axis

from .apply_function_on_2_vectors import apply_function_on_2_vectors


def apply_function_on_vector_and_each_matrix_row(
    target,
    data,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    return apply_along_axis(
        apply_function_on_2_vectors,
        1,
        data,
        target,
        match_function,
        n_required=n_required_for_match_function,
        raise_for_n_less_than_required=raise_for_n_less_than_required,
        raise_for_bad=False,
    )
