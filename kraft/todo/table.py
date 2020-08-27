from numpy import apply_along_axis, full, isnan, nan, unique
from numpy.random import choice, seed
from pandas import DataFrame, Index, Series, concat, isna

from .array import (
    function_on_1_number_array_not_nan,
    guess_type,
    log,
    map_integer,
    normalize as array_normalize,
    shift_minimum,
)
from .CONSTANT import RANDOM_SEED
from .support import cast_builtin
from .table import binarize, drop_axes_label, drop_axis_label, summarize


def normalize_d(dataframe_number, axis, method, **keyword_arguments):

    matrix, axis_0_labels, axis_1_labels, axis_0_name, axis_1_name = untangle(
        dataframe_number
    )

    if axis is None:

        matrix = function_on_1_number_array_not_nan(
            matrix.ravel(), array_normalize, method, update=True, **keyword_arguments
        ).reshape(matrix.shape)

    else:

        matrix = apply_along_axis(
            function_on_1_number_array_not_nan,
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


def normalize_s(vector, method, **normalize_keyword_arguments):

    return Series(
        data=function_on_1_number_array_not_nan(
            vector.to_numpy(),
            array_normalize,
            method,
            update=True,
            **normalize_keyword_arguments,
        ),
        index=vector.index,
        name=vector.name,
    )


def binarize(series):

    object_to_i = {}

    i = 0

    for object_ in series:

        if not isna(object_) and object_ not in object_to_i:

            object_to_i[object_] = i

            i += 1

    object_x_label = full((len(object_to_i), series.size), 0)

    for label_i, object_ in enumerate(series):

        if not isna(object_):

            object_x_label[object_to_i[object_], label_i] = 1

    dataframe = DataFrame(
        data=object_x_label, index=list(object_to_i), columns=series.index,
    )

    dataframe.index.name = series.name

    return dataframe


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


# TODO: check bad index
# TODO: check index.name
def collapse(matrix):

    print(matrix.shape)

    print("Collapsing...")

    matrix = matrix.groupby(level=0).median()

    print(matrix.shape)

    return matrix


def separate_type(feature_x_, drop_constant=True, prefix_feature=True):

    continuous = []

    binary = []

    for _, row in feature_x_.iterrows():

        try:

            is_continuous = (
                guess_type(row.dropna().astype(float).to_numpy()) == "continuous"
            )

        except ValueError:

            is_continuous = False

        if is_continuous:

            continuous.append(row.apply(cast_builtin))

        elif not (drop_constant and row.unique().size == 1):

            binary_x_ = binarize(row)

            if prefix_feature:

                template = "{}.{{}}".format(binary_x_.index.name)

            else:

                template = "{}"

            binary_x_.index = (
                template.format(value) for value in binary_x_.index.to_numpy()
            )

            binary.append(binary_x_)

    template = "{} ({{}})".format(feature_x_.index.name)

    if 0 < len(continuous):

        continuous_x_ = DataFrame(data=continuous)

        continuous_x_.index.name = template.format("continuous")

    else:

        continuous_x_ = None

    if 0 < len(binary):

        binary_x_ = concat(binary)

        binary_x_.index.name = template.format("binary")

    else:

        binary_x_ = None

    return continuous_x_, binary_x_


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
