from numpy import nan


def get_dataframe_fraction_good_on_axis(dataframe, axis, bads=()):

    if 0 < len(bads):

        dataframe = dataframe.replace(bads, nan)

    return 1 - dataframe.isna().sum(axis=axis) / dataframe.shape[axis]
