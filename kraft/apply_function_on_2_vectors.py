from numpy import nan

from .check_array_for_bad import check_array_for_bad


def apply_function_on_2_vectors(
    vector_0,
    vector_1,
    function,
    n_required=None,
    raise_for_n_less_than_n_required=True,
    raise_for_bad=True,
    use_only_good=True,
):

    is_good_0 = ~check_array_for_bad(vector_0, raise_for_bad=raise_for_bad)

    is_good_1 = ~check_array_for_bad(vector_1, raise_for_bad=raise_for_bad)

    if use_only_good:

        is_good = is_good_0 & is_good_1

        if n_required is not None:

            if n_required <= 1:

                n_required *= is_good.size

            n_good = is_good.sum()

            if n_good < n_required:

                if raise_for_n_less_than_n_required:

                    raise ValueError(f"{n_good} <= n_required ({n_required})")

                else:

                    return nan

        vector_0 = vector_0[is_good]

        vector_1 = vector_1[is_good]

    return function(vector_0, vector_1)
