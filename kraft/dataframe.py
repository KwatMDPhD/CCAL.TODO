from numpy import full, nan, unique
from numpy.random import choice
from pandas import DataFrame, concat

from .array import guess_type
from .dataframe import drop_slice
from .plot import plot_heat_map, plot_histogram
from .series import binarize
from .string import BAD_STR
from .support import cast_builtin


def tidy(dataframe):

    assert not dataframe.index.hasnans

    assert not dataframe.index.has_duplicates

    assert not dataframe.columns.hasnans

    assert not dataframe.columns.has_duplicates

    return dataframe.sort_index().sort_index(axis=1)


def drop_slice(
    dataframe,
    axis,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
):

    shape_before = dataframe.shape

    dropped = full(shape_before[axis], False)

    if axis == 0:

        axis_ = 1

    elif axis == 1:

        axis_ = 0

    if max_na is not None:

        if max_na < 1:

            max_n_na = max_na * dataframe.shape[axis_]

        else:

            max_n_na = max_na

        dropped |= dataframe.apply(
            lambda series: max_n_na < series.isna().sum(), axis=axis_
        )

    if min_n_not_na_value is not None:

        dropped |= dataframe.apply(
            lambda series: (~series.isna()).sum() < min_n_not_na_value, axis=axis_
        )

    if min_n_not_na_unique_value is not None:

        dropped |= dataframe.apply(
            lambda series: series[~series.isna()].unique().size
            < min_n_not_na_unique_value,
            axis=axis_,
        )

    if axis == 0:

        dataframe = dataframe.loc[~dropped]

    elif axis == 1:

        dataframe = dataframe.loc[:, ~dropped]

    print(
        "Shape: {} =(drop: axis={}, max_na={}, min_n_not_na_value={}, min_n_not_na_unique_value={})=> {}".format(
            shape_before,
            axis,
            max_na,
            min_n_not_na_value,
            min_n_not_na_unique_value,
            dataframe.shape,
        )
    )

    return dataframe


def drop_slice_greedily(
    dataframe,
    axis=None,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
):

    shape_before = dataframe.shape

    if axis is None:

        axis = int(dataframe.shape[0] < dataframe.shape[1])

    return_ = False

    while True:

        dataframe = drop_slice(
            dataframe,
            axis,
            max_na=max_na,
            min_n_not_na_value=min_n_not_na_value,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
        )

        shape_after = dataframe.shape

        if return_ and shape_before == shape_after:

            return dataframe

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        return_ = True


def group(dataframe):

    print("Grouping index with median...")

    print(dataframe.shape)

    if dataframe.shape[0] == 0:

        return dataframe

    dataframe = dataframe.groupby(level=0).median()

    print(dataframe.shape)

    return dataframe


def make_axis_same(dataframes, axis):

    if axis == 0:

        elements = dataframes[0].index

    else:

        elements = dataframes[0].columns

    for dataframe in dataframes[1:]:

        if axis == 0:

            elements &= dataframe.index

        else:

            elements &= dataframe.columns

    elements = elements.sort_values()

    print("Keeping {} axis-{} elements...".format(elements.size, axis))

    if axis == 0:

        dataframes = tuple(dataframe.loc[elements] for dataframe in dataframes)

    else:

        dataframes = tuple(dataframe[elements] for dataframe in dataframes)

    return dataframes


def make_axis_different(dataframes, axis):

    elements = []

    for dataframe in dataframes:

        if axis == 0:

            elements += dataframe.index.tolist()

        else:

            elements += dataframe.columns.tolist()

    elements, counts = unique(elements, return_counts=True)

    elements_to_drop = elements[1 < counts]

    print("Dropping {} axis-{} elements...".format(elements_to_drop.size, axis))

    return tuple(
        dataframe.drop(elements_to_drop, axis=axis, errors="ignore")
        for dataframe in dataframes
    )


def summarize(
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

            print("Sampling random {} for histogram...".format(plot_histogram_max_size))

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
            layout={"xaxis": {"title": {"text": "Not-NA Value"}}},
        )

    dataframe_isna = dataframe.isna()

    n_na = dataframe_isna.values.sum()

    if 0 < n_na:

        axis0_n_na = dataframe_isna.sum(axis=1)

        axis0_n_na.name = dataframe_isna.index.name

        if axis0_n_na.name is None:

            axis0_n_na.name = "Axis 0"

        axis1_n_na = dataframe_isna.sum()

        axis1_n_na.name = dataframe_isna.columns.name

        if axis1_n_na.name is None:

            axis1_n_na.name = "Axis 1"

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


def separate_type(information_x_, bad_values=BAD_STR):

    continuous_dataframe_rows = []

    binary_dataframes = []

    for information, series in information_x_.iterrows():

        series = series.replace(bad_values, nan)

        if 1 < series.dropna().unique().size:

            try:

                is_continuous = guess_type(series.astype(float)) == "continuous"

            except ValueError:

                is_continuous = False

            if is_continuous:

                continuous_dataframe_rows.append(series.apply(cast_builtin))

            else:

                binary_x_ = binarize(series)

                binary_x_ = binary_x_.loc[~binary_x_.index.isna()]

                binary_x_.index = (
                    "({}) {}".format(binary_x_.index.name, str_)
                    for str_ in binary_x_.index
                )

                binary_dataframes.append(binary_x_)

    continuous_x_ = DataFrame(data=continuous_dataframe_rows)

    continuous_x_.index.name = information_x_.index.name

    binary_x_ = concat(binary_dataframes)

    binary_x_.index.name = information_x_.index.name

    return continuous_x_, binary_x_
