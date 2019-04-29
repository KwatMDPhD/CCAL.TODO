from .drop_dataframe_slice import drop_dataframe_slice


def drop_dataframe_slice_greedily(
    dataframe, max_na=None, min_n_not_na_unique_value=None
):

    modular_shift = int(dataframe.shape[1] < dataframe.shape[0])

    for i in range(dataframe.size):

        shape_before = dataframe.shape

        axis = (i + modular_shift) % 2

        dataframe = drop_dataframe_slice(
            dataframe,
            axis,
            max_na=max_na,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
        )

        shape_after = dataframe.shape

        print("Shape: {} =(drop axis {})=> {}".format(shape_before, axis, shape_after))

        if 0 < i and shape_before == shape_after:

            return dataframe
