from numpy import full, isnan, nan


def ignore_nan_and_function_1(
    array, function, *function_arguments, update=False, **function_keyword_arguments
):

    is_not_nan = ~isnan(array)

    output = function(
        array[is_not_nan], *function_arguments, **function_keyword_arguments
    )

    if update:

        array_ = full(array.shape, nan)

        array_[is_not_nan] = output

        return array_

    else:

        return output
