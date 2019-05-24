from numpy import arange

from .plot_and_save import plot_and_save


def select_series_indices(
    series,
    direction,
    thresholds=None,
    n=None,
    fraction=None,
    standard_deviation=None,
    plot=True,
    title=None,
    xaxis=None,
    yaxis=None,
    html_file_path=None,
):

    series_no_na_sorted = series.dropna().sort_values()

    if series_no_na_sorted.empty:

        raise ValueError("Series has only na.")

    if n is not None:

        if direction in ("<", ">"):

            n = min(n, series_no_na_sorted.size)

        elif direction == "<>":

            n = min(n, series_no_na_sorted.size // 2)

    if fraction is not None:

        if direction in ("<", ">"):

            fraction = min(fraction, 1)

        elif direction == "<>":

            fraction = min(fraction, 0.5)

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

                thresholds = (series_no_na_sorted.iloc[n], series_no_na_sorted.iloc[-n])

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

                thresholds = (None, None)

        is_selected = (series_no_na_sorted <= thresholds[0]) | (
            thresholds[1] <= series_no_na_sorted
        )

    if plot:

        if series_no_na_sorted.size < 1e3:

            mode = "markers"

        else:

            mode = "lines"

        plot_and_save(
            {
                "layout": {"title": title, "xaxis": xaxis, "yaxis": yaxis},
                "data": [
                    {
                        "type": "scatter",
                        "name": "All",
                        "x": arange(series_no_na_sorted.size),
                        "y": series_no_na_sorted,
                        "text": series_no_na_sorted.index,
                        "mode": mode,
                        "marker": {"color": "#d0d0d0"},
                    },
                    {
                        "type": "scatter",
                        "name": "Selected",
                        "x": is_selected.values.nonzero()[0],
                        "y": series_no_na_sorted[is_selected],
                        "text": series_no_na_sorted.index[is_selected],
                        "mode": "markers",
                        "marker": {"color": "#20d9ba"},
                    },
                ],
            },
            html_file_path,
        )

    return series_no_na_sorted.index[is_selected]
