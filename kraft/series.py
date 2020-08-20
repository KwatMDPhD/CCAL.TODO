from numpy import full, quantile
from pandas import DataFrame, Series, isna

from .array import ignore_nan_and_function_1, normalize as array_normalize
from .dict_ import merge
from .plot import plot_plotly


def get_extreme_labels(
    series,
    direction,
    low_and_high=None,
    n=None,
    standard_deviation=None,
    plot=True,
    layout=None,
    file_path=None,
):

    series = series.dropna().sort_values()

    vector = series.to_numpy()

    labels = series.index.to_numpy()

    if low_and_high is None:

        if n is not None:

            if n < 1:

                low = quantile(vector, n)

                high = quantile(vector, 1 - n)

            else:

                n = min(vector.size, n)

                low = vector[n - 1]

                high = vector[-n]

        elif standard_deviation is not None:

            mean = vector.mean()

            margin = vector.std() * standard_deviation

            low = mean - margin

            high = mean + margin

    else:

        low, high = low_and_high

    if direction == "<>":

        is_extreme = (vector <= low) | (high <= vector)

    elif direction == "<":

        is_extreme = vector <= low

    elif direction == ">":

        is_extreme = high <= vector

    labels_extreme = labels[is_extreme]

    if plot:

        base = {
            "xaxis": {"title": {"text": "Rank"}},
            "yaxis": {"title": {"text": series.name}},
        }

        if layout is None:

            layout = base

        else:

            layout = merge(base, layout)

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "name": "All ({})".format(labels.size),
                        "x": labels,
                        "y": vector,
                        "text": labels,
                        "marker": {"color": "#d0d0d0"},
                    },
                    {
                        "name": "Extreme ({})".format(labels_extreme.size),
                        "x": labels_extreme,
                        "y": vector[is_extreme],
                        "text": labels_extreme,
                        "mode": "markers",
                    },
                ],
            },
            file_path=file_path,
        )

    return labels_extreme


def normalize(vector, method, **normalize_keyword_arguments):

    return Series(
        data=ignore_nan_and_function_1(
            vector.to_numpy(),
            array_normalize,
            method,
            update=True,
            **normalize_keyword_arguments
        ),
        index=vector.index,
        name=vector.name,
    )


def binarize(series):

    object_to_i = {}

    i = 0

    for object_ in series:

        if not isna(object_) and object_ not in object_to_i:

            object_to_i[object_] = i

            i += 1

    object_x_label = full((len(object_to_i), series.size), 0)

    for label_i, object_ in enumerate(series):

        if not isna(object_):

            object_x_label[object_to_i[object_], label_i] = 1

    dataframe = DataFrame(
        data=object_x_label, index=list(object_to_i), columns=series.index,
    )

    dataframe.index.name = series.name

    return dataframe
