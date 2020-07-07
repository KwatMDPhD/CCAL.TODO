from math import floor

from numpy import (
    apply_along_axis,
    arange,
    asarray,
    concatenate,
    full,
    isnan,
    median,
    nan,
    unique,
)
from numpy.random import choice
from pandas import DataFrame, Series, isna

from .array import normalize as array_normalize
from .grid import make_grid_nd
from .iterable import map_int
from .plot import plot_heat_map, plot_histogram


def error_axes(dataframe):

    for axis_labels in (dataframe.index, dataframe.columns):

        labels, counts = unique(axis_labels, return_counts=True)

        is_na = isna(labels)

        assert not is_na.any()

        assert (counts == 1).all()


def sync_axis(dataframes, axis, method):

    if method == "union":

        dataframe0 = dataframes[0]

        if axis == 0:

            labels = set(dataframe0.index)

        else:

            labels = set(dataframe0.columns)

        for dataframe in dataframes[1:]:

            if axis == 0:

                labels = labels.union(set(dataframe.index))

            else:

                labels = labels.union(set(dataframe.columns))

        labels = asarray(sorted(labels))

    elif method == "intersection":

        dataframe0 = dataframes[0]

        if axis == 0:

            labels = dataframe0.index.to_list()

        else:

            labels = dataframe0.columns.to_list()

        for dataframe in dataframes[1:]:

            if axis == 0:

                labels += dataframe.index.to_list()

            else:

                labels += dataframe.columns.to_list()

        labels, counts = unique(labels, return_counts=True)

        labels = labels[counts == len(dataframes)]

    print(labels.size)

    return tuple(dataframe.reindex(labels, axis=axis) for dataframe in dataframes)


def drop_axis_label(dataframe, axis, min_good_value=None, min_good_unique_value=None):

    assert min_good_value is not None or min_good_unique_value is not None

    shape_before = dataframe.shape

    is_kept = full(shape_before[axis], True)

    if axis == 0:

        axis_apply = 1

    elif axis == 1:

        axis_apply = 0

    matrix = dataframe.to_numpy()

    if min_good_value is not None:

        if min_good_value < 1:

            min_good_value = min_good_value * shape_before[axis_apply]

        is_kept &= apply_along_axis(
            lambda vector: min_good_value <= (~isna(vector)).sum(), axis_apply, matrix
        )

    if min_good_unique_value is not None:

        if min_good_unique_value < 1:

            min_good_unique_value = min_good_unique_value * dataframe.shape[axis_apply]

        is_kept &= apply_along_axis(
            lambda vector: min_good_unique_value <= unique(vector[~isna(vector)]).size,
            axis_apply,
            matrix,
        )

    if axis == 0:

        dataframe = dataframe.loc[is_kept, :]

    elif axis == 1:

        dataframe = dataframe.loc[:, is_kept]

    print("{} ==> {}".format(shape_before, dataframe.shape))

    return dataframe


def drop_axes_label(
    dataframe, axis=None, min_good_value=None, min_good_unique_value=None
):

    shape_before = dataframe.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    can_return = False

    while True:

        dataframe = drop_axis_label(
            dataframe,
            axis,
            min_good_value=min_good_value,
            min_good_unique_value=min_good_unique_value,
        )

        shape_after = dataframe.shape

        if can_return and shape_before == shape_after:

            return dataframe

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        can_return = True


def pivot(dataframe, axis0, axis1, values, function=None):

    axis0_labels = unique(dataframe[axis0].to_numpy())

    axis1_labels = unique(dataframe[axis1].to_numpy())

    axis0_label_to_i = map_int(axis0_labels)[0]

    axis1_label_to_i = map_int(axis1_labels)[0]

    matrix = full((axis0_labels.size, axis1_labels.size), nan)

    for axis0_label, axis1_label, value in dataframe[[axis0, axis1, values]].to_numpy():

        axis0_i = axis0_label_to_i[axis0_label]

        axis1_i = axis1_label_to_i[axis1_label]

        value_now = matrix[axis0_i, axis1_i]

        if isnan(value_now):

            matrix[axis0_i, axis1_i] = value

        else:

            matrix[axis0_i, axis1_i] = function(value_now, value)

    return DataFrame(matrix, index=axis0_labels, columns=axis1_labels)


def normalize(dataframe, axis, method, **normalize_keyword_arguments):

    matrix = dataframe.to_numpy()

    if axis is None:

        matrix = array_normalize(matrix, method, **normalize_keyword_arguments)

    else:

        matrix = apply_along_axis(
            array_normalize, axis, matrix, method, **normalize_keyword_arguments
        )

    return DataFrame(matrix, index=dataframe.index, columns=dataframe.columns)


#
def sample(dataframe, axis0_size, axis1_size):

    assert axis0_size is not None or axis1_size is not None

    if axis0_size is not None and axis1_size is not None:

        return dataframe.loc[
            choice(
                dataframe.index,
                size=int(floor(dataframe.shape[0] * axis0_size)),
                replace=False,
            ),
            choice(
                dataframe.columns,
                size=int(floor(dataframe.shape[1] * axis1_size)),
                replace=False,
            ),
        ]

    elif axis0_size is not None:

        return dataframe.loc[
            choice(
                dataframe.index,
                size=int(floor(dataframe.shape[0] * axis0_size)),
                replace=False,
            ),
        ]

    else:

        return dataframe[
            choice(
                dataframe.columns,
                size=int(floor(dataframe.shape[1] * axis1_size)),
                replace=False,
            )
        ]


def summarize(
    dataframe,
    plot=True,
    plot_heat_map_max_size=int(1e6),
    plot_histogram_max_size=int(1e3),
):

    if dataframe.index.name is None:

        dataframe.index.name = "Axis 0"

    if dataframe.columns.name is None:

        dataframe.columns.name = "Axis 1"

    print(dataframe.shape)

    size = dataframe.size

    if plot and size <= plot_heat_map_max_size:

        plot_heat_map(dataframe)

    values = dataframe.to_numpy().flatten()

    labels = asarray(
        tuple(
            "{}_{}".format(axis0_label, axis1_label)
            for axis0_label, axis1_label in make_grid_nd(
                (dataframe.index.to_numpy(), dataframe.columns.to_numpy())
            )
        )
    )

    is_good = ~isnan(values)

    values = values[is_good]

    labels = labels[is_good]

    print("Not-NaN min: {:.2e}".format(values.min()))

    print("Not-NaN median: {:.2e}".format(median(values)))

    print("Not-NaN mean: {:.2e}".format(values.mean()))

    print("Not-NaN max: {:.2e}".format(values.max()))

    if plot:

        if plot_histogram_max_size < values.size:

            print("Choosing {} for histogram...".format(plot_histogram_max_size))

            is_chosen = concatenate(
                (
                    choice(
                        arange(values.size), size=plot_histogram_max_size, replace=False
                    ),
                    (values.argmin(), values.argmax()),
                )
            )

            values = values[is_chosen]

            labels = labels[is_chosen]

        plot_histogram(
            (Series(values, index=labels),),
            layout={"xaxis": {"title": {"text": "Not-NaN Value"}}},
        )

    is_na = dataframe.isna().to_numpy()

    n_na = is_na.sum()

    if 0 < n_na:

        if plot:

            plot_histogram(
                (
                    Series(is_na.sum(axis=1), name=dataframe.index.name),
                    Series(is_na.sum(axis=0), name=dataframe.columns.name),
                ),
                layout={
                    "title": {"text": "Fraction NA: {:.2e}".format(n_na / size)},
                    "xaxis": {"title": {"text": "N NA"}},
                },
            )
