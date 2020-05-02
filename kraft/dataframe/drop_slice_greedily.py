from .drop_slice import drop_slice


def drop_slice_greedily(
    dataframe,
    axis=None,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
):

    shape_before = dataframe.shape

    if axis is None:

        axis = int(dataframe.shape[0] < dataframe.shape[1])

    return_ = False

    while True:

        dataframe = drop_slice(
            dataframe,
            axis,
            max_na=max_na,
            min_n_not_na_value=min_n_not_na_value,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
        )

        shape_after = dataframe.shape

        if return_ and shape_before == shape_after:

            return dataframe

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        return_ = True
