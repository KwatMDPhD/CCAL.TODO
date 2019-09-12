from math import ceil

from numpy import apply_along_axis, array_split, concatenate, full, nan
from numpy.random import choice, get_state, seed, set_state
from pandas import DataFrame

from .apply_function_on_permuted_vector_and_each_matrix_row import (
    apply_function_on_permuted_vector_and_each_matrix_row,
)
from .apply_function_on_vector_and_each_matrix_row import (
    apply_function_on_vector_and_each_matrix_row,
)
from .call_function_with_multiprocess import call_function_with_multiprocess
from .check_array_for_bad import check_array_for_bad
from .compute_empirical_p_values_and_fdrs import compute_empirical_p_values_and_fdrs
from .compute_information_coefficient_between_2_vectors import (
    compute_information_coefficient_between_2_vectors,
)
from .compute_normal_pdf_margin_of_error import compute_normal_pdf_margin_of_error
from .RANDOM_SEED import RANDOM_SEED
from .select_series_indices import select_series_indices


def apply_function_on_vector_and_each_matrix_row_and_compute_statistics(
    vector,
    matrix,
    n_job=1,
    match_function=compute_information_coefficient_between_2_vectors,
    n_required_for_match_function=2,
    raise_for_n_less_than_required=False,
    n_extreme=8,
    fraction_extreme=None,
    random_seed=RANDOM_SEED,
    n_sampling=10,
    n_permutation=10,
    score_ascending=False,
):

    score_moe_p_value_fdr = DataFrame(
        index=range(matrix.shape[0]), columns=("Score", "0.95 MoE", "P-Value", "FDR")
    )

    n_job = min(matrix.shape[0], n_job)

    print("Computing score with {} ({} job)...".format(match_function.__name__, n_job))

    matrix_split = array_split(matrix, n_job)

    scores = concatenate(
        call_function_with_multiprocess(
            apply_function_on_vector_and_each_matrix_row,
            (
                (
                    vector,
                    matrix_,
                    match_function,
                    n_required_for_match_function,
                    raise_for_n_less_than_required,
                )
                for matrix_ in matrix_split
            ),
            n_job,
        )
    )

    if check_array_for_bad(scores, raise_for_bad=False).all():

        raise

    score_moe_p_value_fdr["Score"] = scores

    if n_extreme is not None or fraction_extreme is not None:

        moe_indices = select_series_indices(
            score_moe_p_value_fdr["Score"],
            "<>",
            n=n_extreme,
            fraction=fraction_extreme,
            plot=False,
        )

    else:

        moe_indices = score_moe_p_value_fdr.index

    print("Computing MoE with {} sampling...".format(n_sampling))

    seed(seed=random_seed)

    index_x_sampling = full((moe_indices.size, n_sampling), nan)

    n_sample = ceil(0.632 * vector.size)

    for i in range(n_sampling):

        random_indices = choice(vector.size, size=n_sample, replace=True)

        random_state = get_state()

        index_x_sampling[:, i] = apply_function_on_vector_and_each_matrix_row(
            vector[random_indices],
            matrix[moe_indices][:, random_indices],
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    score_moe_p_value_fdr.loc[moe_indices, "0.95 MoE"] = apply_along_axis(
        compute_normal_pdf_margin_of_error, 1, index_x_sampling, raise_for_bad=False
    )

    p_values, fdrs = compute_empirical_p_values_and_fdrs(
        score_moe_p_value_fdr["Score"],
        concatenate(
            call_function_with_multiprocess(
                apply_function_on_permuted_vector_and_each_matrix_row,
                (
                    (
                        vector,
                        matrix_,
                        random_seed,
                        n_permutation,
                        match_function,
                        n_required_for_match_function,
                        raise_for_n_less_than_required,
                    )
                    for matrix_ in matrix_split
                ),
                n_job,
            )
        ).flatten(),
        "<>",
        raise_for_bad=False,
    )

    score_moe_p_value_fdr["P-Value"] = p_values

    score_moe_p_value_fdr["FDR"] = fdrs

    return score_moe_p_value_fdr.sort_values("Score", ascending=score_ascending)
