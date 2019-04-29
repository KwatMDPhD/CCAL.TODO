from numpy import asarray


def drop_dataframe_slice(dataframe, axis, max_na=None, min_n_not_na_unique_value=None):

    dropped = asarray((False,) * dataframe.shape[(axis + 1) % 2])

    if max_na is not None:

        if max_na < 1:

            max_n_na = max_na * dataframe.shape[axis]

        else:

            max_n_na = max_na

        dropped |= dataframe.apply(
            lambda series: max_n_na < series.isna().sum(), axis=axis
        )

    if min_n_not_na_unique_value is not None:

        dropped |= dataframe.apply(
            lambda series: series[~series.isna()].unique().size
            < min_n_not_na_unique_value,
            axis=axis,
        )

    if axis == 0:

        return dataframe.loc[:, ~dropped]

    elif axis == 1:

        return dataframe.loc[~dropped]
