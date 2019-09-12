from .check_array_for_bad import check_array_for_bad


def check_dataframe_number(dataframe):

    if dataframe.index.has_duplicates:

        raise

    if dataframe.columns.has_duplicates:

        raise

    if not dataframe.applymap(
        lambda value: isinstance(value, (int, float))
    ).values.all():

        raise

    check_array_for_bad(dataframe.values)
