from .check_array_for_bad import check_array_for_bad


def check_dataframe_number(dataframe):

    assert not dataframe.index.has_duplicates

    assert not dataframe.columns.has_duplicates

    assert dataframe.applymap(
        lambda value: isinstance(value, (int, float))
    ).values.all()

    check_array_for_bad(dataframe.values)
