from numpy import full


def drop_dataframe_slice(
    dataframe,
    axis,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
):

    shape_before = dataframe.shape

    dropped = full(shape_before[axis], False)

    axis_for_applying = (axis + 1) % 2

    if max_na is not None:

        if max_na < 1:

            max_n_na = max_na * dataframe.shape[axis_for_applying]

        else:

            max_n_na = max_na

        dropped |= dataframe.apply(
            lambda series: max_n_na < series.isna().sum(), axis=axis_for_applying
        )

    if min_n_not_na_value is not None:

        dropped |= dataframe.apply(
            lambda series: series[~series.isna()].size < min_n_not_na_value,
            axis=axis_for_applying,
        )

    if min_n_not_na_unique_value is not None:

        dropped |= dataframe.apply(
            lambda series: series[~series.isna()].unique().size
            < min_n_not_na_unique_value,
            axis=axis_for_applying,
        )

    if axis == 0:

        dataframe = dataframe.loc[~dropped]

    elif axis == 1:

        dataframe = dataframe.loc[:, ~dropped]

    print(
        "Shape: {} =(drop: axis={}, max_na={}, min_n_not_na_value={}, min_n_not_na_unique_value={})=> {}".format(
            shape_before,
            axis,
            max_na,
            min_n_not_na_value,
            min_n_not_na_unique_value,
            dataframe.shape,
        )
    )

    return dataframe
