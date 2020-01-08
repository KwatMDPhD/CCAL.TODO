from numpy import nan


def ignore_nan_and_function(
    vector_0, vector_1, function, **function_keyword_arguments,
):

    is_not_nan = (vector_0 != nan) & (vector_1 != nan)

    return function(
        vector_0[is_not_nan], vector_1[is_not_nan], **function_keyword_arguments
    )
