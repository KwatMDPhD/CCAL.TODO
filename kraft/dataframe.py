from numpy import (
    apply_along_axis,
    asarray,
    concatenate,
    full,
    isnan,
    ix_,
    median,
    nan,
    unique,
)
from numpy.random import choice, seed
from pandas import DataFrame, Index, Series, isna

from .array import ignore_nan_and_function_1, map_int, normalize as array_normalize
from .CONSTANT import RANDOM_SEED
from .grid import make_grid_nd
from .plot import plot_heat_map, plot_histogram


def error_axes(dataframe):

    for labels in dataframe.axes:

        labels, counts = unique(labels.to_numpy(), return_counts=True)

        is_na = isna(labels)

        assert not is_na.any()

        assert (counts == 1).all()


def peak(dataframe, n_axis_0=4, n_axis_1=2):

    print("-" * 80)

    print(dataframe.iloc[:n_axis_0, :n_axis_1])

    print(dataframe.shape)

    print("-" * 80)


def print_value_n(dataframe, axis):

    assert axis in (0, 1)

    if axis == 0:

        generator = dataframe.iterrows()

    else:

        generator = dataframe.items()

    for label, values in generator:

        value_n = values.value_counts()

        print("=" * 80)

        print(value_n)


def sample(
    dataframe,
    axis_0_n,
    axis_1_n,
    random_seed=RANDOM_SEED,
    axis_0_keyword_arguments=None,
    axis_1_keyword_arguments=None,
):

    matrix = dataframe.to_numpy()

    axis_0_size, axis_1_size = matrix.shape

    seed(seed=random_seed)

    if axis_0_n is not None:

        if axis_0_n < 1:

            axis_0_n = int(axis_0_n * axis_0_size)

        if axis_0_keyword_arguments is None:

            axis_0_keyword_arguments = {}

        axis_0_is = choice(axis_0_size, size=axis_0_n, **axis_0_keyword_arguments)

    if axis_1_n is not None:

        if axis_1_n < 1:

            axis_1_n = int(axis_1_n * axis_1_size)

        if axis_1_keyword_arguments is None:

            axis_1_keyword_arguments = {}

        axis_1_is = choice(axis_1_size, size=axis_1_n, **axis_1_keyword_arguments)

    if axis_0_n is not None and axis_1_n is not None:

        return DataFrame(
            data=matrix[ix_(axis_0_is, axis_1_is)],
            index=dataframe.index[axis_0_is],
            columns=dataframe.columns[axis_1_is],
        )

    elif axis_0_n is not None:

        return DataFrame(
            data=matrix[axis_0_is, :],
            index=dataframe.index[axis_0_is],
            columns=dataframe.columns,
        )

    elif axis_1_n is not None:

        return DataFrame(
            data=matrix[:, axis_1_is],
            index=dataframe.index,
            columns=dataframe.columns[axis_1_is],
        )


def drop_axis_label(
    dataframe, axis, min_n_not_na_value=None, min_n_not_na_unique_value=None
):

    assert min_n_not_na_value is not None or min_n_not_na_unique_value is not None

    shape_before = dataframe.shape

    is_kept = full(shape_before[axis], True)

    if axis == 0:

        axis_ = 1

    elif axis == 1:

        axis_ = 0

    matrix = dataframe.to_numpy()

    if min_n_not_na_value is not None:

        if min_n_not_na_value < 1:

            min_n_not_na_value = min_n_not_na_value * shape_before[axis_]

        def function_0(vector):

            return min_n_not_na_value <= (~isna(vector)).sum()

        is_kept &= apply_along_axis(function_0, axis_, matrix)

    if min_n_not_na_unique_value is not None:

        if min_n_not_na_unique_value < 1:

            min_n_not_na_unique_value = (
                min_n_not_na_unique_value * dataframe.shape[axis_]
            )

        def function_1(vector):

            return min_n_not_na_unique_value <= unique(vector[~isna(vector)]).size

        is_kept &= apply_along_axis(function_1, axis_, matrix,)

    if axis == 0:

        dataframe = dataframe.loc[is_kept, :]

    elif axis == 1:

        dataframe = dataframe.loc[:, is_kept]

    print("{} ==> {}".format(shape_before, dataframe.shape))

    return dataframe


def drop_axes_label(
    dataframe, axis=None, min_n_not_na_value=None, min_n_not_na_unique_value=None
):

    shape_before = dataframe.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    can_return = False

    while True:

        dataframe = drop_axis_label(
            dataframe,
            axis,
            min_n_not_na_value=min_n_not_na_value,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
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


def sync_axis(dataframes, axis):

    dataframe_0 = dataframes[0]

    labels = dataframe_0.axes[axis]

    for dataframe in dataframes[1:]:

        labels = labels.union(set(dataframe.axes[axis]))

    return tuple(
        dataframe.reindex(labels=asarray(sorted(labels)), axis=axis)
        for dataframe in dataframes
    )


def normalize(matrix, axis, method, **keyword_arguments):

    axis_0, axis_1 = matrix.axes

    matrix = matrix.to_numpy()

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
            **keyword_arguments
        )

    return DataFrame(data=matrix, index=axis_0, columns=axis_1)


def summarize(
    matrix,
    plot=True,
    plot_heat_map_max_size=int(1e6),
    plot_histogram_max_size=int(1e3),
):

    if matrix.index.name is None:

        matrix.index.name = "Axis 0"

    if matrix.columns.name is None:

        matrix.columns.name = "Axis 1"

    print(matrix.shape)

    matrix_size = matrix.size

    if plot and matrix_size <= plot_heat_map_max_size:

        plot_heat_map(matrix)

    axis_0, axis_1 = matrix.axes

    matrix = matrix.to_numpy()

    is_nan = isnan(matrix)

    n_nan = is_nan.sum()

    if 0 < n_nan:

        if plot:

            plot_histogram(
                (
                    Series(data=is_nan.sum(axis=1), index=axis_0, name="N NaN"),
                    Series(data=is_nan.sum(axis=0), index=axis_1),
                ),
                layout={
                    "title": {"text": "% NaN: {:.2%}".format(n_nan / matrix_size)},
                },
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
                for label_0, label_1 in make_grid_nd((axis_0, axis_1))[~is_nan.ravel()]
            )
        )

        if plot_histogram_max_size < not_nan_numbers.size:

            print("Choosing {} for histogram...".format(plot_histogram_max_size))

            is_chosen = concatenate(
                (
                    choice(
                        not_nan_numbers.size,
                        size=plot_histogram_max_size,
                        replace=False,
                    ),
                    (not_nan_numbers.argmin(), not_nan_numbers.argmax()),
                )
            )

            not_nan_numbers = not_nan_numbers[is_chosen]

            labels = labels[is_chosen]

        plot_histogram(
            (Series(data=not_nan_numbers, index=labels, name="(Not-NaN) Number"),),
        )

        plot_histogram(
            (
                Series(
                    data=median(matrix, axis=1), index=axis_0, name="(Not-NaN) Median"
                ),
                Series(data=median(matrix, axis=0), index=axis_1),
            ),
        )


def pivot(
    axis_0, axis_1, values, function=None, axis_0_name="Axis 0", axis_1_name="Axis 1",
):

    axis_0_label_to_i = map_int(axis_0)[0]

    axis_1_label_to_i = map_int(axis_1)[0]

    matrix = full((len(axis_0_label_to_i), len(axis_1_label_to_i)), nan)

    for axis_0_label, axis_1_label, value in zip(axis_0, axis_1, values):

        axis_0_i = axis_0_label_to_i[axis_0_label]

        axis_1_i = axis_1_label_to_i[axis_1_label]

        value_now = matrix[axis_0_i, axis_1_i]

        if isnan(value_now) or function is None:

            matrix[axis_0_i, axis_1_i] = value

        else:

            matrix[axis_0_i, axis_1_i] = function(value_now, value)

    return DataFrame(
        data=matrix,
        index=Index(axis_0_label_to_i, name=axis_0_name),
        columns=Index(axis_1_label_to_i, name=axis_1_name),
    )
