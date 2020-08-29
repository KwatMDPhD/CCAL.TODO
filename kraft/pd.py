from numpy import asarray, concatenate, full, isnan, logical_not, median
from numpy.random import choice
from pandas import DataFrame, Index, concat, isna

from .array import guess_type
from .grid import make_nd
from .plot import plot_heat_map, plot_histogram
from .support import cast_builtin

# ==============================================================================
# Series
# ==============================================================================


def binarize(sr):

    value_to_axis_0_index = {}

    axis_0_index = 0

    for value in sr:

        if not isna(value) and value not in value_to_axis_0_index:

            value_to_axis_0_index[value] = axis_0_index

            axis_0_index += 1

    value_x_label = full((len(value_to_axis_0_index), sr.size), 0)

    for axis_1_index, value in enumerate(sr):

        if not isna(value):

            value_x_label[value_to_axis_0_index[value], axis_1_index] = 1

    dataframe = DataFrame(
        data=value_x_label,
        index=Index(data=value_to_axis_0_index, name=sr.name),
        columns=sr.index,
    )

    return dataframe


# ==============================================================================
# DataFrame
# ==============================================================================


def peek(df, axis_0_axis_n=4, axis_1_label_n=2):

    print("-" * 80)

    print(df.iloc[:axis_0_axis_n, :axis_1_label_n])

    print(df.shape)

    print("-" * 80)


def sync(df_, axis):

    # TODO: refactor

    df_0 = df_[0]

    label_ = df_0.axes[axis]

    for df in df_[1:]:

        label_ = label_.intersection(df.axes[axis])

    label_ = asarray(sorted(label_))

    return tuple(df.reindex(labels=label_, axis=axis) for df in df_)


# ==============================================================================
# Number DataFrame
# ==============================================================================


def summarize(
    number_df,
    plot=True,
    df_name="DataFrame Name",
    heat_map_max_size=int(1e6),
    histogram_max_size=int(1e3),
):

    matrix = number_df.to_numpy()

    axis_0_label_ = number_df.index.to_numpy()

    axis_1_label_ = number_df.columns.to_numpy()

    axis_0_name = number_df.index.name

    axis_1_name = number_df.columns.name

    print(matrix.shape)

    title = {"text": df_name}

    matrix_size = matrix.size

    if plot and matrix_size <= heat_map_max_size:

        plot_heat_map(
            matrix,
            axis_0_label_,
            axis_1_label_,
            axis_0_name,
            axis_1_name,
            layout={"title": title},
        )

    is_nan_matrix = isnan(matrix)

    nan_n = is_nan_matrix.sum()

    if 0 < nan_n:

        print("% NaN: {:.2%}".format(nan_n / matrix_size))

        if plot:

            plot_histogram(
                (is_nan_matrix.sum(axis=1), is_nan_matrix.sum(axis=0)),
                (axis_0_label_, axis_1_label_),
                (axis_0_name, axis_1_name),
                layout={"title": title, "xaxis": {"title": {"text": "N NaN"}}},
            )

    is_not_nan_matrix = logical_not(is_nan_matrix)

    matrix_not_nan = matrix[is_not_nan_matrix].ravel()

    print("(Not-NaN) min: {:.2e}".format(matrix_not_nan.min()))

    print("(Not-NaN) median: {:.2e}".format(median(matrix_not_nan)))

    print("(Not-NaN) mean: {:.2e}".format(matrix_not_nan.mean()))

    print("(Not-NaN) max: {:.2e}".format(matrix_not_nan.max()))

    if plot:

        label_ = asarray(
            tuple(
                "{}_{}".format(label_0, label_1)
                for label_0, label_1 in make_nd((axis_0_label_, axis_1_label_))[
                    is_not_nan_matrix.ravel()
                ]
            )
        )

        if histogram_max_size < matrix_not_nan.size:

            print("Choosing {} for histogram...".format(histogram_max_size))

            index_ = concatenate(
                (
                    choice(
                        matrix_not_nan.size, size=histogram_max_size, replace=False,
                    ),
                    (matrix_not_nan.argmin(), matrix_not_nan.argmax()),
                )
            )

            matrix_not_nan = matrix_not_nan[index_]

            label_ = label_[index_]

        plot_histogram(
            (matrix_not_nan,),
            (label_,),
            ("All",),
            layout={"title": title, "xaxis": {"title": {"text": "(Not-NaN) Number"}}},
        )

        plot_histogram(
            (median(matrix, axis=1), median(matrix, axis=0)),
            (axis_0_label_, axis_1_label_),
            (axis_0_name, axis_1_name),
            layout={"title": title, "xaxis": {"title": {"text": "(Not-NaN) Median"}}},
        )


def collapse(number_df):

    print(number_df.shape)

    print("Collapsing...")

    number_df = number_df.groupby(level=0).median()

    print(number_df.shape)

    return number_df


# ==============================================================================
# Feature_x_Sample
# ==============================================================================


def separate_type(feature_x_sample, drop_constant=True, prefix_feature=True):

    continuous_row_ = []

    binary_x_sample_ = []

    for _, row in feature_x_sample.iterrows():

        try:

            is_continuous = (
                guess_type(row.dropna().astype(float).to_numpy()) == "continuous"
            )

        except ValueError:

            is_continuous = False

        if is_continuous:

            continuous_row_.append(row.apply(cast_builtin))

        elif not (drop_constant and row.unique().size == 1):

            binary_x_sample = binarize(row)

            if prefix_feature:

                axis_0_label_template = "{}.{{}}".format(binary_x_sample.index.name)

            else:

                axis_0_label_template = "{}"

            binary_x_sample.index = (
                axis_0_label_template.format(label)
                for label in binary_x_sample.index.to_numpy()
            )

            binary_x_sample_.append(binary_x_sample)

    axis_0_name_template = "{} ({{}})".format(feature_x_sample.index.name)

    if 0 < len(continuous_row_):

        continuous_x_sample = DataFrame(data=continuous_row_)

        continuous_x_sample.index.name = axis_0_name_template.format("continuous")

    else:

        continuous_x_sample = None

    if 0 < len(binary_x_sample_):

        binary_x_sample = concat(binary_x_sample_)

        binary_x_sample.index.name = axis_0_name_template.format("binary")

    else:

        binary_x_sample = None

    return continuous_x_sample, binary_x_sample


# TODO
from numpy import apply_along_axis, full, isnan, nan, unique
from numpy.random import choice, seed
from pandas import DataFrame, Index, isna

from .array import (
    function_on_1_number_array_not_nan,
    guess_type,
    log,
    map_integer,
    normalize as array_normalize,
    shift_minimum,
)
from .CONSTANT import RANDOM_SEED
from .table import drop_axes_label, drop_axis_label, summarize


def error_axes(dataframe):

    for axis in dataframe.axes:

        axis, counts = unique(axis.to_numpy(), return_counts=True)

        is_na = isna(axis)

        assert not is_na.any()

        assert (counts == 1).all()


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


def pivot(
    axis_0_labels, axis_1_labels, numbers, axis_0_name, axis_1_name, function=None,
):

    axis_0_label_to_i = map_integer(axis_0_labels)[0]

    axis_1_label_to_i = map_integer(axis_1_labels)[0]

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


def process(
    feature_x_sample,
    features_to_drop=(),
    samples_to_drop=(),
    nanize=None,
    good_axis=None,
    min_good_value=None,
    min_good_unique_value=None,
    log_shift=None,
    log_base=None,
    normalize_axis=None,
    normalize_method=None,
    clip_min=None,
    clip_max=None,
    **keyword_arguments,
):

    if 0 < len(features_to_drop):

        print(
            "Dropping {}: {}...".format(feature_x_sample.index.name, features_to_drop)
        )

        feature_x_sample.drop(labels=features_to_drop, errors="ignore", inplace=True)

        summarize(feature_x_sample, **keyword_arguments)

    if 0 < len(samples_to_drop):

        print(
            "Dropping {}: {}...".format(feature_x_sample.columns.name, samples_to_drop)
        )

        feature_x_sample.drop(
            labels=samples_to_drop, axis=1, errors="ignore", inplace=True
        )

        summarize(feature_x_sample, **keyword_arguments)

    if nanize is not None:

        print("NaNizing <= {}...".format(nanize))

        matrix = feature_x_sample.to_numpy()

        matrix[matrix <= nanize] = nan

        feature_x_sample = DataFrame(
            data=matrix, index=feature_x_sample.index, columns=feature_x_sample.columns
        )

        summarize(feature_x_sample, **keyword_arguments)

    if min_good_value is not None or min_good_unique_value is not None:

        print("Dropping slice...")

        if good_axis is None:

            drop_function = drop_axes_label

        else:

            drop_function = drop_axis_label

        shape = feature_x_sample.shape

        feature_x_sample = drop_function(
            feature_x_sample,
            good_axis,
            min_n_not_na_value=min_good_value,
            min_n_not_na_unique_value=min_good_unique_value,
        )

        if shape != feature_x_sample.shape:

            summarize(feature_x_sample, **keyword_arguments)

    if log_base is not None:

        print(
            "Logging (shift_before_log={}, log_base={})...".format(log_shift, log_base)
        )

        matrix = feature_x_sample.to_numpy()

        if log_shift is not None:

            matrix = shift_minimum(matrix, log_shift)

        feature_x_sample = DataFrame(
            data=log(matrix, log_base=log_base,),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **keyword_arguments)

    if normalize_method is not None:

        print("Axis-{} {} normalizing...".format(normalize_axis, normalize_method))

        feature_x_sample = normalize(feature_x_sample, normalize_axis, normalize_method)

        summarize(feature_x_sample, **keyword_arguments)

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        feature_x_sample.clip(lower=clip_min, upper=clip_max, inplace=True)

        summarize(feature_x_sample, **keyword_arguments)

    return feature_x_sample
