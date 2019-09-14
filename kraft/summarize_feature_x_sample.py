from numpy import nanmax, nanmin
from numpy.random import choice

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_feature_x_sample(
    feature_x_sample,
    feature_x_sample_alias="Feature-x-Sample",
    feature_x_sample_value_name="Feature-x-Sample Value",
    plot=True,
    plot_heat_map_max_size=1e6,
    plot_histogram_max_size=1e5,
):

    print("Shape: {}".format(feature_x_sample.shape))

    print("Not-NaN min: {:.2e}".format(nanmin(feature_x_sample.values)))

    print("Not-NaN max: {:.2e}".format(nanmax(feature_x_sample.values)))

    if plot:

        if feature_x_sample.size <= plot_heat_map_max_size:

            plot_heat_map(
                feature_x_sample, layout={"title": {"text": feature_x_sample_alias}}
            )

        feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

        feature_x_sample_not_na_values.name = feature_x_sample_value_name

        if plot_histogram_max_size < feature_x_sample_not_na_values.size:

            print(
                "Sampling random {} values for histogram...".format(
                    plot_histogram_max_size
                )
            )

            feature_x_sample_not_na_values = feature_x_sample_not_na_values[
                choice(
                    feature_x_sample_not_na_values.index,
                    size=plot_histogram_max_size,
                    replace=False,
                )
            ]

        plot_histogram(
            (feature_x_sample_not_na_values,),
            plot_rug=False,
            layout={
                "title": {"text": feature_x_sample_alias},
                "xaxis": {"title": {"text": "Not-NA Value"}},
            },
        )

    isna__feature_x_sample = feature_x_sample.isna()

    n_na = isna__feature_x_sample.values.sum()

    print("Fraction NA: {:.2e}".format(n_na / feature_x_sample.size))

    if n_na and plot and isna__feature_x_sample.size <= plot_histogram_max_size:

        plot_histogram(
            (isna__feature_x_sample.sum(axis=1), isna__feature_x_sample.sum()),
            layout={
                "title": {"text": feature_x_sample_alias},
                "xaxis": {"title": {"text": "N NA"}},
            },
        )
