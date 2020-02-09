from .merge_2_dicts import merge_2_dicts
from .plot_plotly import plot_plotly


def select(
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

    if thresholds is None:

        if n is not None:

            n = min(n, series_no_na_sorted.size)

            threshold_low = series_no_na_sorted.iloc[n - 1]

            threshold_high = series_no_na_sorted.iloc[-n]

        elif fraction is not None:

            threshold_low = series_no_na_sorted.quantile(fraction)

            threshold_high = series_no_na_sorted.quantile(1 - fraction)

        elif standard_deviation is not None:

            mean = series_no_na_sorted.mean()

            margin = series_no_na_sorted.std() * standard_deviation

            threshold_low = mean - margin

            threshold_high = mean + margin

    else:

        threshold_low, threshold_high = thresholds

    if direction == "<>":

        is_selected = (series_no_na_sorted <= threshold_low) | (
            threshold_high <= series_no_na_sorted
        )

    elif direction == "<":

        is_selected = series_no_na_sorted <= threshold_low

    elif direction == ">":

        is_selected = threshold_high <= series_no_na_sorted

    index = series_no_na_sorted.index[is_selected]

    if plot:

        layout_ = {
            "title": {"text": index.name},
            "xaxis": {"title": {"text": "Rank"}},
        }

        if layout is None:

            layout = layout_

        else:

            layout = merge_2_dicts(layout_, layout)

        n_selected = is_selected.sum()

        if n_selected < 64:

            mode = "markers+text"

        elif n_selected < 256:

            mode = "markers"

        else:

            mode = "lines"

        plot_plotly(
            {
                "layout": layout,
                "data": [
                    {
                        "name": "All ({})".format(series_no_na_sorted.size),
                        "x": series_no_na_sorted.index,
                        "y": series_no_na_sorted,
                        "text": series_no_na_sorted.index,
                        "marker": {"color": "#d0d0d0"},
                    },
                    {
                        "name": "Selected ({})".format(n_selected),
                        "x": index,
                        "y": series_no_na_sorted[index],
                        "text": index,
                        "mode": mode,
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return index
