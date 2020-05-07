from numpy import isnan


def ignore_nan_and_function_2(
    array_0, array_1, function, *function_arguments, **function_keyword_arguments,
):

    is_not_nan = ~isnan(array_0) & ~isnan(array_1)

    return function(
        array_0[is_not_nan],
        array_1[is_not_nan],
        *function_arguments,
        **function_keyword_arguments,
    )
