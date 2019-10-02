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

    if feature_x_sample.size <= plot_heat_map_max_size:

        plot_heat_map(
            feature_x_sample, layout={"title": {"text": feature_x_sample_name}}
        )

    feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

    feature_x_sample_not_na_values.name = feature_x_sample_value_name

    print("Not-NA min: {:.2e}".format(feature_x_sample_not_na_values.min()))

    print("Not-NA median: {:.2e}".format(feature_x_sample_not_na_values.median()))

    print("Not-NA mean: {:.2e}".format(feature_x_sample_not_na_values.mean()))

    print("Not-NA max: {:.2e}".format(feature_x_sample_not_na_values.max()))

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

    if 0 < n_na:

        feature_n_na = feature_x_sample_isna.sum(axis=1)

        feature_n_na.name = feature_x_sample_isna.index.name

        sample_n_na = feature_x_sample_isna.sum()

        sample_n_na.name = feature_x_sample_isna.columns.name

        plot_histogram(
            (feature_n_na, sample_n_na),
            plot_rug=False,
            layout={
                "title": {
                    "text": "{}<br>Fraction NA: {:.2e}".format(
                        feature_x_sample_name, n_na / feature_x_sample.size
                    )
                },
                "xaxis": {"title": {"text": "N NA"}},
            },
        )
