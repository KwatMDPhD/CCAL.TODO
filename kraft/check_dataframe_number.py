from .check_array_for_bad import check_array_for_bad


def check_dataframe_number(dataframe):

    if dataframe.index.has_duplicates:

        raise ValueError

    if dataframe.columns.has_duplicates:

        raise ValueError

    if not dataframe.applymap(
        lambda value: isinstance(value, (int, float))
    ).values.all():

        raise ValueError

    check_array_for_bad(dataframe.values)
