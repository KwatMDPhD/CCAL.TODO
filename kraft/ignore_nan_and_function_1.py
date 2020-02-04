from numpy import full, isnan, nan


def ignore_nan_and_function_1(
    array, function, *function_arguments, **function_keyword_arguments
):

    array_ = full(array.shape, nan)

    is_not_nan = ~isnan(array)

    array_[is_not_nan] = function(
        array[is_not_nan], *function_arguments, **function_keyword_arguments
    )

    return array_
