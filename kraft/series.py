from numpy import full, quantile
from pandas import DataFrame, isna

from .plot import plot_plotly
from .support import merge_2_dicts


def select_extreme(
    series,
    direction,
    thresholds=None,
    n=None,
    fraction=None,
    standard_deviation=None,
    plot=True,
    layout=None,
    html_file_path=None,
):

    series = series.dropna().sort_values()

    labels = series.index.to_numpy()

    vector = series.to_numpy()

    if thresholds is None:

        if n is not None:

            n = min(vector.size, n)

            threshold_low = vector[n - 1]

            threshold_high = vector[-n]

        elif fraction is not None:

            threshold_low = quantile(vector, fraction)

            threshold_high = quantile(vector, 1 - fraction)

        elif standard_deviation is not None:

            mean = vector.mean()

            margin = vector.std() * standard_deviation

            threshold_low = mean - margin

            threshold_high = mean + margin

    else:

        threshold_low, threshold_high = thresholds

    if direction == "<>":

        is_selected = (vector <= threshold_low) | (threshold_high <= vector)

    elif direction == "<":

        is_selected = vector <= threshold_low

    elif direction == ">":

        is_selected = threshold_high <= vector

    labels_selected = labels[is_selected]

    if plot:

        layout_template = {
            "xaxis": {"title": {"text": "Rank"}},
            "yaxis": {"title": {"text": series.name}},
        }

        if layout is None:

            layout = layout_template

        else:

            layout = merge_2_dicts(layout_template, layout)

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
                        "name": "Selected ({})".format(is_selected.sum()),
                        "x": labels_selected,
                        "y": vector[is_selected],
                        "text": labels_selected,
                        "mode": "markers",
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return labels_selected


def binarize(series):

    object_to_i = {}

    i = 0

    for object_ in series:

        if not (isna(object_) or object_ in object_to_i):

            object_to_i[object_] = i

            i += 1

    object_x_label = full((len(object_to_i), series.size), 0)

    for i, object_ in enumerate(series):

        if not isna(object_):

            object_x_label[object_to_i[object_], i] = 1

    dataframe = DataFrame(
        object_x_label, index=list(object_to_i), columns=series.index,
    )

    dataframe.index.name = series.name

    return dataframe
