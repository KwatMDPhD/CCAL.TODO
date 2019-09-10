from numpy import nanmax, nanmin
from numpy.random import choice
from pandas import Series

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_feature_x_sample(
    feature_x_sample,
    feature_x_sample_alias="Feature-x-Sample",
    plot=True,
    plot_heat_map_max_size=1e6,
    plot_histogram_max_size=1e5,
    plot_rug_max_size=1e4,
):

    print("Shape: {}".format(feature_x_sample.shape))

    print("Not-NaN min: {}".format(nanmin(feature_x_sample.values)))

    print("Not-NaN max: {}".format(nanmax(feature_x_sample.values)))

    if plot:

        if feature_x_sample.size <= plot_heat_map_max_size:

            plot_heat_map(
                feature_x_sample,
                layout={
                    "title": {"text": feature_x_sample_alias},
                    "xaxis": {"title": {"text": feature_x_sample.columns.name}},
                    "yaxis": {"title": {"text": feature_x_sample.index.name}},
                },
            )

        feature_x_sample_not_na_values = feature_x_sample.unstack().dropna()

        if plot_histogram_max_size < feature_x_sample_not_na_values.size:

            print(
                "Sampling random {} values for histogram...".format(
                    plot_histogram_max_size
                )
            )

            feature_x_sample_not_na_values = choice(
                feature_x_sample_not_na_values,
                size=plot_histogram_max_size,
                replace=False,
            )

        plot_histogram(
            (Series(feature_x_sample_not_na_values),),
            plot_rug=feature_x_sample_not_na_values.size <= plot_rug_max_size,
            layout={"title": {"text": feature_x_sample_alias}},
            xaxis={"title": {"text": "Not-NA Value"}},
        )

    isna__feature_x_sample = feature_x_sample.isna()

    n_na = isna__feature_x_sample.values.sum()

    print("% NA: {:.3f}".format(n_na / feature_x_sample.size * 100))

    if n_na and plot and isna__feature_x_sample.size <= plot_histogram_max_size:

        plot_histogram(
            (isna__feature_x_sample.sum(axis=1), isna__feature_x_sample.sum()),
            plot_rug=max(isna__feature_x_sample.shape) <= plot_rug_max_size,
            layout={"title": {"text": feature_x_sample_alias}},
            xaxis={"title": {"text": "N NA"}},
        )
