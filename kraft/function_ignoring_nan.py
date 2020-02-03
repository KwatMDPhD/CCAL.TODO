from numpy import isnan


def function_ignoring_nan(
    vector_0, vector_1, function, **function_keyword_arguments,
):

    is_not_nan = ~isnan(vector_0) & ~isnan(vector_1)

    return function(
        vector_0[is_not_nan], vector_1[is_not_nan], **function_keyword_arguments
    )
