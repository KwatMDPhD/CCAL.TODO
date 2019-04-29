from pandas import DataFrame


def make_binary_dataframe_from_categorical_series(series):

    object_x_index = DataFrame(
        index=series.dropna().sort_values().unique(), columns=series.index
    )

    object_x_index.index.name = series.name

    for object in object_x_index.index:

        object_x_index.loc[object] = (series == object).astype(int)

    return object_x_index
