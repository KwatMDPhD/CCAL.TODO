def tidy(dataframe):

    assert not dataframe.index.hasnans

    assert not dataframe.index.has_duplicates

    assert not dataframe.columns.hasnans

    assert not dataframe.columns.has_duplicates

    return dataframe.sort_index().sort_index(axis=1)
