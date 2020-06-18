from pandas import DataFrame

from .plot import plot_plotly
from .RANDOM_SEED import RANDOM_SEED
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
                        "mode": "markers",
                    },
                ],
            },
            html_file_path=html_file_path,
        )

    return index


def binarize(series):
    dataframe = DataFrame(index=series.unique(), columns=series.index)

    dataframe.index.name = series.name

    for str_ in dataframe.index:

        dataframe.loc[str_] = series == str_

    return dataframe.astype(int)


def sample_from_each_series_value(series, n_per_value=None, random_seed=RANDOM_SEED):

    if n_per_value is None:

        n_per_value = series.value_counts().min()

        print("n_per_value = {}".format(n_per_value))

    indices_selected = []

    for group_name, group_series in series.groupby(by=series):

        if n_per_value <= group_series.size:

            indices_selected.extend(
                group_series.sample(
                    n=n_per_value, random_state=random_seed
                ).index.sort_values()
            )

        else:

            print(
                "There is not enough {} to sample without replacement.".format(
                    group_name
                )
            )

    return series[indices_selected]
