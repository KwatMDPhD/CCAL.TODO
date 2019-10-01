from numpy import median, nanmax, nanmin
from numpy.random import choice

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_feature_x_sample(
    feature_x_sample,
    feature_x_sample_name="Feature-x-Sample",
    feature_x_sample_value_name="Feature-x-Sample Value",
    plot_heat_map_max_size=1e6,
    plot_histogram_max_size=1e3,
):

    print("Shape: {}".format(feature_x_sample.shape))

    print("Not-NaN min: {:.2e}".format(nanmin(feature_x_sample.values)))

    print("Not-NaN median: {:.2e}".format(median(feature_x_sample.values)))

    print("Not-NaN mean: {:.2e}".format(feature_x_sample.values.mean()))

    print("Not-NaN max: {:.2e}".format(nanmax(feature_x_sample.values)))

    if feature_x_sample.size <= plot_heat_map_max_size:

        plot_heat_map(
            feature_x_sample, layout={"title": {"text": feature_x_sample_name}}
        )

    feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

    feature_x_sample_not_na_values.name = feature_x_sample_value_name

    if plot_histogram_max_size < feature_x_sample_not_na_values.size:

        print(
            "Sampling random {:.2e} {} for histogram...".format(
                plot_histogram_max_size, feature_x_sample_value_name
            )
        )

        feature_x_sample_not_na_values = feature_x_sample_not_na_values[
            choice(
                feature_x_sample_not_na_values.index,
                size=int(plot_histogram_max_size),
                replace=False,
            )
        ]

    plot_histogram(
        (feature_x_sample_not_na_values,),
        plot_rug=False,
        layout={
            "title": {"text": feature_x_sample_name},
            "xaxis": {"title": {"text": "Not-NA Value"}},
        },
    )

    feature_x_sample_isna = feature_x_sample.isna()

    n_na = feature_x_sample_isna.values.sum()

    print("Fraction NA: {:.2e}".format(n_na / feature_x_sample.size))

    if 0 < n_na:

        feature_n_na = feature_x_sample_isna.sum(axis=1)

        feature_n_na.name = feature_x_sample_isna.index.name

        sample_n_na = feature_x_sample_isna.sum()

        sample_n_na.name = feature_x_sample_isna.columns.name

        plot_histogram(
            (feature_n_na, sample_n_na),
            layout={
                "title": {"text": feature_x_sample_name},
                "xaxis": {"title": {"text": "N NA"}},
            },
        )
