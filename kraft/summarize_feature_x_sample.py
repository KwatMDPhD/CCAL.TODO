from numpy import nanmax, nanmin
from numpy.random import choice
from pandas import Series

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_feature_x_sample(
    feature_x_sample,
    feature_x_sample_alias="Feature-x-Sample",
    feature_x_sample_value_name="Value",
    plot=True,
    plot_heat_map_max_size=160000,
    plot_histogram_max_size=16000,
    plot_rug_max_size=1600,
):

    print(f"Shape: {feature_x_sample.shape}")

    print(f"(not-nan) Min: {nanmin(feature_x_sample.values)}")

    print(f"(not-nan) Max: {nanmax(feature_x_sample.values)}")

    for axis in (0, 1):

        n_unique_value_count = feature_x_sample.apply(
            lambda series: series.unique().size, axis=axis
        ).value_counts()

        n_unique_value_count.index.name = "N Unique"

        n_unique_value_count.name = "Count"

        n_unique_value_count = n_unique_value_count.to_frame()

        n_extreme = 8

        if n_extreme * 2 < n_unique_value_count.shape[0]:

            print(f"Axis {axis} Top and Bottom {n_extreme} Number of Unique Values:")

            print(
                n_unique_value_count.iloc[
                    list(range(n_extreme)) + list(range(-n_extreme, 0))
                ]
            )

        else:

            print(f"Axis {axis} Number of Unique Values:")

            print(n_unique_value_count)

    if plot:

        if feature_x_sample.size < plot_heat_map_max_size:

            plot_heat_map(
                feature_x_sample,
                title=feature_x_sample_alias,
                xaxis_title=feature_x_sample.columns.name,
                yaxis_title=feature_x_sample.index.name,
            )

        feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

        if plot_histogram_max_size < feature_x_sample_not_na_values.size:

            print(f"Sampling random {plot_histogram_max_size:,} values ...")

            feature_x_sample_not_na_values = choice(
                feature_x_sample_not_na_values,
                size=plot_histogram_max_size,
                replace=False,
            )

        value_name = "Not-NA Value"

        plot_histogram(
            (Series(feature_x_sample_not_na_values),),
            plot_rug=feature_x_sample_not_na_values.size < plot_rug_max_size,
            title=f"{feature_x_sample_alias}<br>Histogram of {value_name}",
            xaxis_title=value_name,
        )

    isna__feature_x_sample = feature_x_sample.isna()

    n_na = isna__feature_x_sample.values.sum()

    print(f"N NA: {n_na} ({n_na / feature_x_sample.size * 100:.2f}%)")

    if n_na and plot and isna__feature_x_sample.size < plot_histogram_max_size:

        value_name = "N NA"

        plot_histogram(
            (isna__feature_x_sample.sum(axis=1), isna__feature_x_sample.sum()),
            plot_rug=max(isna__feature_x_sample.shape) < plot_rug_max_size,
            title=f"{feature_x_sample_alias}<br>Histogram of {value_name} ",
            xaxis_title=value_name,
        )
