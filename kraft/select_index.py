from numpy import arange

from .merge_2_dicts import merge_2_dicts


def select_index(
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

    series_no_na_sorted = series.dropna().sort_values()

    if n is not None:

        if direction in ("<", ">"):

            n = min(n, series_no_na_sorted.size)

        elif direction == "<>":

            n = min(n, series_no_na_sorted.size // 2)

    elif fraction is not None:

        if direction in ("<", ">"):

            fraction = min(fraction, 1)

        elif direction == "<>":

            fraction = min(fraction, 1 / 2)

    if direction == "<":

        if thresholds is None:

            if n is not None:

                threshold = series_no_na_sorted.iloc[n]

            elif fraction is not None:

                threshold = series_no_na_sorted.quantile(fraction)

            elif standard_deviation is not None:

                threshold = (
                    series_no_na_sorted.mean()
                    - series_no_na_sorted.std() * standard_deviation
                )

            else:

                threshold = None

        else:

            threshold = thresholds[0]

        is_selected = series_no_na_sorted <= threshold

    elif direction == ">":

        if thresholds is None:

            if n is not None:

                threshold = series_no_na_sorted.iloc[-n]

            elif fraction is not None:

                threshold = series_no_na_sorted.quantile(1 - fraction)

            elif standard_deviation is not None:

                threshold = (
                    series_no_na_sorted.mean()
                    + series_no_na_sorted.std() * standard_deviation
                )

            else:

                threshold = None

        else:

            threshold = thresholds[0]

        is_selected = threshold <= series_no_na_sorted

    elif direction == "<>":

        if thresholds is None:

            if n is not None:

                thresholds = (
                    series_no_na_sorted.iloc[n - 1],
                    series_no_na_sorted.iloc[-n],
                )

            elif fraction is not None:

                thresholds = (
                    series_no_na_sorted.quantile(fraction),
                    series_no_na_sorted.quantile(1 - fraction),
                )

            elif standard_deviation is not None:

                thresholds = (
                    series_no_na_sorted.mean()
                    - series_no_na_sorted.std() * standard_deviation,
                    series_no_na_sorted.mean()
                    + series_no_na_sorted.std() * standard_deviation,
                )

            else:

                thresholds = (None,) * 2

        is_selected = (series_no_na_sorted <= thresholds[0]) | (
            thresholds[1] <= series_no_na_sorted
        )

    selected_index = series_no_na_sorted.index[is_selected]

    if plot:

        layout_template = {
            "title": {"text": series_no_na_sorted.index.name},
            "xaxis": {"title": {"text": "Rank"}},
        }

        if layout is None:

            layout = layout_template

        else:

            layout = merge_2_dicts(layout_template, layout)

        n_selected = is_selected.sum()

        if n_selected < 64:

            mode = "markers+text"

        elif n_selected < 256:

            mode = "markers"

        else:

            mode = "lines"

        plot(
            {
                "layout": layout,
                "data": [
                    {
                        "type": "scatter",
                        "name": "All ({})".format(series_no_na_sorted.size),
                        "x": arange(series_no_na_sorted.size),
                        "y": series_no_na_sorted,
                        "text": series_no_na_sorted.index,
                        "marker": {"color": "#d0d0d0"},
                    },
                    {
                        "type": "scatter",
                        "name": "Selected ({})".format(n_selected),
                        "x": is_selected.values.nonzero()[0],
                        "y": series_no_na_sorted[is_selected],
                        "text": selected_index,
                        "mode": mode,
                    },
                ],
            },
            html_file_path,
        )

    return selected_index
