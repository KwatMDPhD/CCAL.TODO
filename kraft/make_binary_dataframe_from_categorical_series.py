from pandas import DataFrame


def make_binary_dataframe_from_categorical_series(
    series, include_series_name_in_index=False
):

    object_x_index = DataFrame(
        index=series.dropna().sort_values().unique(), columns=series.index
    )

    for object_ in object_x_index.index:

        object_x_index.loc[object_] = (series == object_).astype(int)

    object_x_index.index.name = series.name

    object_x_index.index = (
        "({}) {}".format(series.name, object_) for object_ in object_x_index.index
    )

    return object_x_index
