from numpy import full, nan
from numpy.random import get_state, seed, set_state, shuffle

from .apply_function_on_vector_and_each_matrix_row import (
    apply_function_on_vector_and_each_matrix_row,
)


def apply_function_on_permuted_vector_and_each_matrix_row(
    target,
    data,
    random_seed,
    n_permutation,
    match_function,
    n_required_for_match_function,
    raise_for_n_less_than_required,
):

    print("Computing p-value and FDR with {} permutation...".format(n_permutation))

    seed(seed=random_seed)

    index_x_permutation = full((data.shape[0], n_permutation), nan)

    target_ = target.copy()

    for i in range(n_permutation):

        shuffle(target_)

        random_state = get_state()

        index_x_permutation[:, i] = apply_function_on_vector_and_each_matrix_row(
            target_,
            data,
            match_function,
            n_required_for_match_function,
            raise_for_n_less_than_required,
        )

        set_state(random_state)

    return index_x_permutation
