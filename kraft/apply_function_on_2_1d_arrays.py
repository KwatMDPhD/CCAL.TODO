from numpy import nan

from .check_nd_array_for_bad import check_nd_array_for_bad


def apply_function_on_2_1d_arrays(
    _1d_array_0,
    _1d_array_1,
    function,
    n_required=None,
    raise_for_n_less_than_required=True,
    raise_for_bad=True,
    use_only_good=True,
):

    is_good_0 = ~check_nd_array_for_bad(_1d_array_0, raise_for_bad=raise_for_bad)

    is_good_1 = ~check_nd_array_for_bad(_1d_array_1, raise_for_bad=raise_for_bad)

    if use_only_good:

        is_good = is_good_0 & is_good_1

        if n_required is not None:

            if n_required <= 1:

                n_required *= is_good.size

            n_good = is_good.sum()
            if n_good < n_required:

                if raise_for_n_less_than_required:

                    raise ValueError(f"{n_good} <= n_required ({n_required})")

                else:

                    return nan

        _1d_array_0 = _1d_array_0[is_good]

        _1d_array_1 = _1d_array_1[is_good]

    return function(_1d_array_0, _1d_array_1)
