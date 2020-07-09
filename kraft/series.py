from numpy import full, quantile
from pandas import DataFrame, isna

from .dict_ import merge
from .plot import plot_plotly


def select_extreme_labels(
    series,
    direction,
    low_and_high=None,
    n=None,
    fraction=None,
    standard_deviation=None,
    plot=True,
    layout=None,
    html_file_path=None,
):

    series = series.dropna().sort_values()

    vector = series.to_numpy()

    labels = series.index.to_numpy()

    if low_and_high is None:

        if n is not None:

            n = min(vector.size, n)

            low = vector[n - 1]

            high = vector[-n]

        elif fraction is not None:

            low = quantile(vector, fraction)

            high = quantile(vector, 1 - fraction)

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

        layout_base = {
            "xaxis": {"title": {"text": "Rank"}},
            "yaxis": {"title": {"text": series.name}},
        }

        if layout is None:

            layout = layout_base

        else:

            layout = merge(layout_base, layout)

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
                        "name": "Selected ({})".format(labels_extreme.size),
                        "x": labels_extreme,
                        "y": vector[is_extreme],
                        "text": labels_extreme,
                        "mode": "markers",
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return labels_extreme


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
        object_x_label, index=list(object_to_i), columns=series.index,
    )

    dataframe.index.name = series.name

    return dataframe
