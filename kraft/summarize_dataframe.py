from numpy.random import choice

from .plot_heat_map import plot_heat_map
from .plot_histogram import plot_histogram


def summarize_dataframe(
    dataframe,
    plot=True,
    plot_heat_map_max_size=int(1e6),
    plot_histogram_max_size=int(1e3),
):

    print("Shape: {}".format(dataframe.shape))

    if plot and dataframe.size <= plot_heat_map_max_size:

        plot_heat_map(dataframe)

    dataframe_not_na_values = dataframe.unstack().dropna()

    print("Not-NA min: {:.2e}".format(dataframe_not_na_values.min()))

    print("Not-NA median: {:.2e}".format(dataframe_not_na_values.median()))

    print("Not-NA mean: {:.2e}".format(dataframe_not_na_values.mean()))

    print("Not-NA max: {:.2e}".format(dataframe_not_na_values.max()))

    if plot:

        if plot_histogram_max_size < dataframe_not_na_values.size:

            print(
                "Sampling random {} values for histogram...".format(
                    plot_histogram_max_size
                )
            )

            dataframe_not_na_values = dataframe_not_na_values[
                choice(
                    dataframe_not_na_values.index,
                    size=plot_histogram_max_size,
                    replace=False,
                ).tolist()
                + [dataframe_not_na_values.idxmin(), dataframe_not_na_values.idxmax()]
            ]

        plot_histogram(
            (dataframe_not_na_values,),
            plot_rug=False,
            layout={"xaxis": {"title": {"text": "Not-NA Value"}}},
        )

    dataframe_isna = dataframe.isna()

    n_na = dataframe_isna.values.sum()

    if 0 < n_na:

        axis0_n_na = dataframe_isna.sum(axis=1)

        axis0_n_na.name = dataframe_isna.index.name

        axis1_n_na = dataframe_isna.sum()

        axis1_n_na.name = dataframe_isna.columns.name

        if plot:

            plot_histogram(
                (axis0_n_na, axis1_n_na),
                layout={
                    "title": {
                        "text": "Fraction NA: {:.2e}".format(n_na / dataframe.size)
                    },
                    "xaxis": {"title": {"text": "N NA"}},
                },
            )
