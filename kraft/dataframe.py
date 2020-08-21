from numpy import (
    apply_along_axis,
    asarray,
    concatenate,
    full,
    isnan,
    median,
    nan,
    unique,
)
from numpy.random import choice, seed
from pandas import DataFrame, Index, isna

from .array import ignore_nan_and_function_1, map_int, normalize as array_normalize
from .CONSTANT import RANDOM_SEED
from .grid import make_grid_nd
from .plot import plot_heat_map, plot_histogram


def error_axes(dataframe):

    for axis in dataframe.axes:

        axis, counts = unique(axis.to_numpy(), return_counts=True)

        is_na = isna(axis)

        assert not is_na.any()

        assert (counts == 1).all()


def peak(dataframe, n_axis_0_label=4, n_axis_1_label=2):

    print("-" * 80)

    print(dataframe.iloc[:n_axis_0_label, :n_axis_1_label])

    print(dataframe.shape)

    print("-" * 80)


def print_value_n(dataframe, axis):

    assert axis in (0, 1)

    if axis == 0:

        generator = dataframe.iterrows()

    else:

        generator = dataframe.items()

    for _, values in generator:

        value_n = values.value_counts()

        print("-" * 80)

        print(value_n)


def untangle(dataframe):

    return (
        dataframe.to_numpy(),
        dataframe.index.to_numpy(),
        dataframe.columns.to_numpy(),
        dataframe.index.name,
        dataframe.columns.name,
    )


def sample(
    dataframe,
    n_axis_0_label,
    n_axis_1_label,
    random_seed=RANDOM_SEED,
    **keyword_arguments,
):

    matrix, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(dataframe)

    axis_0_size = axis_0_labels.size

    axis_1_size = axis_1_labels.size

    seed(seed=random_seed)

    if n_axis_0_label is not None:

        if n_axis_0_label < 1:

            n_axis_0_label = int(n_axis_0_label * axis_0_size)

        if keyword_arguments is None:

            keyword_arguments = {}

        axis_0_is = choice(axis_0_size, size=n_axis_0_label, **keyword_arguments)

    if n_axis_1_label is not None:

        if n_axis_1_label < 1:

            n_axis_1_label = int(n_axis_1_label * axis_1_size)

        if keyword_arguments is None:

            keyword_arguments = {}

        axis_1_is = choice(axis_1_size, size=n_axis_1_label, **keyword_arguments)

    if n_axis_0_label is not None and n_axis_1_label is not None:

        return DataFrame(
            data=matrix[axis_0_is, axis_1_is],
            index=Index(data=axis_0_labels[axis_0_is], name=axis_0_name),
            columns=Index(data=axis_1_labels[axis_1_is], name=axis_1_name),
        )

    elif n_axis_0_label is not None:

        return DataFrame(
            data=matrix[axis_0_is],
            index=Index(data=axis_0_labels[axis_0_is], name=axis_0_name),
            columns=Index(data=axis_1_labels, name=axis_1_name),
        )

    elif n_axis_1_label is not None:

        return DataFrame(
            data=matrix[:, axis_1_is],
            index=Index(data=axis_0_labels, name=axis_0_name),
            columns=Index(data=axis_1_labels[axis_1_is], name=axis_1_name),
        )


def drop_axis_label(
    dataframe, axis, min_not_na_value=None, min_not_na_unique_value=None
):

    assert all(min_not_na_value is not None, min_not_na_unique_value is not None)

    shape_before = dataframe.shape

    is_keep = full(shape_before[axis], True)

    if axis == 0:

        axis_ = 1

    elif axis == 1:

        axis_ = 0

    matrix = dataframe.to_numpy()

    if min_not_na_value is not None:

        if min_not_na_value < 1:

            min_not_na_value = min_not_na_value * shape_before[axis_]

        def function_0(vector):

            return min_not_na_value <= (~isna(vector)).sum()

        is_keep &= apply_along_axis(function_0, axis_, matrix)

    if min_not_na_unique_value is not None:

        if min_not_na_unique_value < 1:

            min_not_na_unique_value = min_not_na_unique_value * dataframe.shape[axis_]

        def function_1(vector):

            return min_not_na_unique_value <= unique(vector[~isna(vector)]).size

        is_keep &= apply_along_axis(function_1, axis_, matrix)

    if axis == 0:

        dataframe = dataframe.loc[is_keep, :]

    elif axis == 1:

        dataframe = dataframe.loc[:, is_keep]

    print("{} ==> {}".format(shape_before, dataframe.shape))

    return dataframe


def drop_axes_label(
    dataframe, axis=None, min_not_na_value=None, min_not_na_unique_value=None
):

    shape_before = dataframe.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    can_return = False

    while True:

        dataframe = drop_axis_label(
            dataframe,
            axis,
            min_not_na_value=min_not_na_value,
            min_not_na_unique_value=min_not_na_unique_value,
        )

        shape_after = dataframe.shape

        if all(can_return, shape_before == shape_after):

            return dataframe

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        can_return = True


def sync_axis(dataframes, axis):

    dataframe_0 = dataframes[0]

    labels = dataframe_0.axes[axis]

    for dataframe in dataframes[1:]:

        labels = labels.union(dataframe.axes[axis])

    labels = asarray(sorted(labels))

    return tuple(
        dataframe.reindex(labels=labels, axis=axis) for dataframe in dataframes
    )


def normalize(dataframe_number, axis, method, **keyword_arguments):

    matrix, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(
        dataframe_number
    )

    if axis is None:

        matrix = ignore_nan_and_function_1(
            matrix.ravel(), array_normalize, method, update=True, **keyword_arguments
        ).reshape(matrix.shape)

    else:

        matrix = apply_along_axis(
            ignore_nan_and_function_1,
            axis,
            matrix,
            array_normalize,
            method,
            update=True,
            **keyword_arguments,
        )

    return DataFrame(
        data=matrix,
        index=Index(data=axis_0_labels, name=axis_0_name),
        columns=Index(data=axis_1_labels, name=axis_1_name),
    )


def summarize(
    dataframe_number,
    plot=True,
    plot_heat_map_max_size=int(1e6),
    plot_histogram_max_size=int(1e3),
):

    matrix, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(
        dataframe_number
    )

    print(matrix.shape)

    matrix_size = matrix.size

    if plot and matrix_size <= plot_heat_map_max_size:

        plot_heat_map(
            matrix, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name,
        )

    is_nan = isnan(matrix)

    n_nan = is_nan.sum()

    if 0 < n_nan:

        print("% NaN: {:.2%}".format(n_nan / matrix_size))

        if plot:

            plot_histogram(
                (is_nan.sum(axis=1), is_nan.sum(axis=0)),
                (axis_0_labels, axis_1_labels),
                (axis_0_name, axis_1_name),
                layout={"xaxis": {"title": {"text": "N NaN"}}},
            )

    not_nan_numbers = matrix[~is_nan].ravel()

    print("(Not-NaN) min: {:.2e}".format(not_nan_numbers.min()))

    print("(Not-NaN) median: {:.2e}".format(median(not_nan_numbers)))

    print("(Not-NaN) mean: {:.2e}".format(not_nan_numbers.mean()))

    print("(Not-NaN) max: {:.2e}".format(not_nan_numbers.max()))

    if plot:

        labels = asarray(
            tuple(
                "{}_{}".format(label_0, label_1)
                for label_0, label_1 in make_grid_nd((axis_0_labels, axis_1_labels))[
                    ~is_nan.ravel()
                ]
            )
        )

        if plot_histogram_max_size < not_nan_numbers.size:

            print("Choosing {} for histogram...".format(plot_histogram_max_size))

            is_ = concatenate(
                (
                    choice(
                        not_nan_numbers.size,
                        size=plot_histogram_max_size,
                        replace=False,
                    ),
                    (not_nan_numbers.argmin(), not_nan_numbers.argmax()),
                )
            )

            not_nan_numbers = not_nan_numbers[is_]

            labels = labels[is_]

        plot_histogram(
            (not_nan_numbers,),
            (labels,),
            ("All",),
            layout={"xaxis": {"title": {"text": "(Not-NaN) Number"}}},
        )

        plot_histogram(
            (median(matrix, axis=1), median(matrix, axis=0)),
            (axis_0_labels, axis_1_labels),
            (axis_0_name, axis_1_name),
            layout={"xaxis": {"title": {"text": "(Not-NaN) Median"}}},
        )


def pivot(
    axis_0_labels, axis_1_labels, numbers, axis_0_name, axis_1_name, function=None,
):

    axis_0_label_to_i = map_int(axis_0_labels)[0]

    axis_1_label_to_i = map_int(axis_1_labels)[0]

    matrix = full((len(axis_0_label_to_i), len(axis_1_label_to_i)), nan)

    for axis_0_label, axis_1_label, number in zip(
        axis_0_labels, axis_1_labels, numbers
    ):

        i_0 = axis_0_label_to_i[axis_0_label]

        i_1 = axis_1_label_to_i[axis_1_label]

        number_now = matrix[i_0, i_1]

        if isnan(number_now) or function is None:

            matrix[i_0, i_1] = number

        else:

            matrix[i_0, i_1] = function(number_now, number)

    return DataFrame(
        data=matrix,
        index=Index(data=axis_0_label_to_i, name=axis_0_name),
        columns=Index(data=axis_1_label_to_i, name=axis_1_name),
    )
