from numpy import (apply_along_axis, asarray, concatenate, full, isnan,
                   logical_not, median, nan, unique)
from numpy.random import choice, seed
from pandas import DataFrame, Index, concat, isna

from .array import (check_is_not_na, guess_type, log, map_int, normalize,
                    shift_min)
from .CONSTANT import RANDOM_SEED
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

    matrix = full((len(value_to_axis_0_index), sr.size), 0)

    for axis_1_index, value in enumerate(sr):

        if not isna(value):

            matrix[value_to_axis_0_index[value], axis_1_index] = 1

    return DataFrame(
        data=matrix,
        index=Index(data=value_to_axis_0_index, name=sr.name),
        columns=sr.index,
    )


# ==============================================================================
# DataFrame
# ==============================================================================


def error_axes(df):

    for label_ in df.axes:

        label_, count_ = unique(label_.to_numpy(), return_counts=True)

        is_not_na_ = check_is_not_na(label_)

        assert is_not_na_.all()

        assert (count_ == 1).all()


def peek(df, axis_0_axis_n=4, axis_1_label_n=2):

    print("-" * 80)

    print(df.iloc[:axis_0_axis_n, :axis_1_label_n])

    print(df.shape)

    print("-" * 80)


def count_value(df):

    for label, value_ in df.items():

        print("-" * 80)

        print(label)

        value_n_ = value_.value_counts()

        print(value_n_)

        print("-" * 80)


def drop_axis_label(df, axis, not_na_min_n=None, not_na_unique_min_n=None):

    assert not_na_min_n is not None and not_na_unique_min_n is not None

    shape_before = df.shape

    is_keep_ = full(shape_before[axis], True)

    if axis == 0:

        apply_axis = 1

    elif axis == 1:

        apply_axis = 0

    matrix = df.to_numpy()

    if not_na_min_n is not None:

        if not_na_min_n < 1:

            not_na_min_n = not_na_min_n * shape_before[apply_axis]

        is_keep_ &= apply_along_axis(
            _check_has_enough_not_na, apply_axis, matrix, not_na_min_n
        )

    if not_na_unique_min_n is not None:

        if not_na_unique_min_n < 1:

            not_na_unique_min_n = not_na_unique_min_n * df.shape[apply_axis]

        is_keep_ &= apply_along_axis(
            _check_has_enough_not_na_unique, apply_axis, matrix, not_na_unique_min_n
        )

    if axis == 0:

        df = df.loc[is_keep_, :]

    elif axis == 1:

        df = df.loc[:, is_keep_]

    print("{} => {}".format(shape_before, df.shape))

    return df


def drop_axes_label(df, axis=None, not_na_min_n=None, not_na_unique_min_n=None):

    shape_before = df.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    can_return = False

    while True:

        df = drop_axis_label(
            df,
            axis,
            not_na_min_n=not_na_min_n,
            not_na_unique_min_n=not_na_unique_min_n,
        )

        shape_after = df.shape

        if can_return and shape_before == shape_after:

            return df

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        can_return = True


def sample(
    df,
    axis_0_label_n,
    axis_1_label_n,
    random_seed=RANDOM_SEED,
    **kwarg_,
):

    axis_0_size, axis_1_size = df.shape

    seed(seed=random_seed)

    if axis_0_label_n is not None:

        if axis_0_label_n < 1:

            axis_0_label_n = int(axis_0_label_n * axis_0_size)

        axis_0_index_ = choice(axis_0_size, size=axis_0_label_n, **kwarg_)

    if axis_1_label_n is not None:

        if axis_1_label_n < 1:

            axis_1_label_n = int(axis_1_label_n * axis_1_size)

        axis_1_index_ = choice(axis_1_size, size=axis_1_label_n, **kwarg_)

    matrix = df.to_numpy()

    axis_0_label_ = df.index.to_numpy()

    axis_1_label_ = df.columns.to_numpy()

    axis_0_name = df.index.name

    axis_1_name = df.columns.name

    if axis_0_label_n is not None and axis_1_label_n is not None:

        return DataFrame(
            data=matrix[axis_0_index_, axis_1_index_],
            index=Index(data=axis_0_label_[axis_0_index_], name=axis_0_name),
            columns=Index(data=axis_1_label_[axis_1_index_], name=axis_1_name),
        )

    elif axis_0_label_n is not None:

        return DataFrame(
            data=matrix[axis_0_index_],
            index=Index(data=axis_0_label_[axis_0_index_], name=axis_0_name),
            columns=Index(data=axis_1_label_, name=axis_1_name),
        )

    elif axis_1_label_n is not None:

        return DataFrame(
            data=matrix[:, axis_1_index_],
            index=Index(data=axis_0_label_, name=axis_0_name),
            columns=Index(data=axis_1_label_[axis_1_index_], name=axis_1_name),
        )


def sync(df_, axis):

    df_0 = df_[0]

    label_ = df_0.axes[axis]

    for df in df_[1:]:

        label_ = label_.intersection(df.axes[axis])

    label_ = asarray(sorted(label_))

    return tuple(df.reindex(labels=label_, axis=axis) for df in df_)


def pivot(
    axis_0_label_,
    axis_1_label_,
    value_,
    axis_0_name,
    axis_1_name,
    function=None,
):

    axis_0_label_to_index = map_int(axis_0_label_)[0]

    axis_1_label_to_index = map_int(axis_1_label_)[0]

    matrix = full((len(axis_0_label_to_index), len(axis_1_label_to_index)), nan)

    for axis_0_label, axis_1_label, value in zip(axis_0_label_, axis_1_label_, value_):

        axis_0_index = axis_0_label_to_index[axis_0_label]

        axis_1_index = axis_1_label_to_index[axis_1_label]

        value_now = matrix[axis_0_index, axis_1_index]

        if isnan(value_now) or function is None:

            matrix[axis_0_index, axis_1_index] = value

        else:

            matrix[axis_0_index, axis_1_index] = function(value_now, value)

    return DataFrame(
        data=matrix,
        index=Index(data=axis_0_label_to_index, name=axis_0_name),
        columns=Index(data=axis_1_label_to_index, name=axis_1_name),
    )


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

    size = matrix.size

    if plot and size <= heat_map_max_size:

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

        print("% NaN: {:.2%}".format(nan_n / size))

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
                        matrix_not_nan.size,
                        size=histogram_max_size,
                        replace=False,
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


# TODO: add options including geometric mean
def collapse(number_df):

    print(number_df.shape)

    print("Collapsing...")

    number_df = number_df.groupby(level=0).median()

    print(number_df.shape)

    return number_df


# ==============================================================================
# Feature x Sample
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

                label_template = "{}.{{}}".format(binary_x_sample.index.name)

            else:

                label_template = "{}"

            binary_x_sample.index = (
                label_template.format(label)
                for label in binary_x_sample.index.to_numpy()
            )

            binary_x_sample_.append(binary_x_sample)

    name_template = "{} ({{}})".format(feature_x_sample.index.name)

    if 0 < len(continuous_row_):

        continuous_x_sample = DataFrame(data=continuous_row_)

        continuous_x_sample.index.name = name_template.format("continuous")

    else:

        continuous_x_sample = None

    if 0 < len(binary_x_sample_):

        binary_x_sample = concat(binary_x_sample_)

        binary_x_sample.index.name = name_template.format("binary")

    else:

        binary_x_sample = None

    return continuous_x_sample, binary_x_sample


def _check_has_enough_not_na(vector, not_na_min_n):

    return not_na_min_n <= check_is_not_na(vector).sum()


def _check_has_enough_not_na_unique(vector, not_na_unique_min_n):

    return not_na_unique_min_n <= unique(vector[check_is_not_na(vector)]).size


def process(
    feature_x_sample,
    drop_feature_=(),
    drop_sample_=(),
    nanize=None,
    drop_axis=None,
    drop_not_na_min_n=None,
    drop_not_na_unique_min_n=None,
    log_shift_min=None,
    log_base=None,
    normalize_axis=None,
    normalize_method=None,
    clip_min=None,
    clip_max=None,
    **kwarg_,
):

    if 0 < len(drop_feature_):

        print("Dropping {}: {}...".format(feature_x_sample.index.name, drop_feature_))

        feature_x_sample = feature_x_sample.drop(labels=drop_feature_, errors="ignore")

        summarize(feature_x_sample, **kwarg_)

    if 0 < len(drop_sample_):

        print("Dropping {}: {}...".format(feature_x_sample.columns.name, drop_sample_))

        feature_x_sample = feature_x_sample.drop(
            labels=drop_sample_, axis=1, errors="ignore"
        )

        summarize(feature_x_sample, **kwarg_)

    if nanize is not None:

        print("NaNizing <= {}...".format(nanize))

        matrix = feature_x_sample.to_numpy()

        matrix[matrix <= nanize] = nan

        feature_x_sample = DataFrame(
            data=matrix, index=feature_x_sample.index, columns=feature_x_sample.columns
        )

        summarize(feature_x_sample, **kwarg_)

    if drop_not_na_min_n is not None or drop_not_na_unique_min_n is not None:

        print("Dropping slice...")

        if drop_axis is None:

            drop_function = drop_axes_label

        else:

            drop_function = drop_axis_label

        shape = feature_x_sample.shape

        feature_x_sample = drop_function(
            feature_x_sample,
            drop_axis,
            not_na_min_n=drop_not_na_min_n,
            not_na_unique_min_n=drop_not_na_unique_min_n,
        )

        if shape != feature_x_sample.shape:

            summarize(feature_x_sample, **kwarg_)

    if log_base is not None:

        print("Logging (log_min={}, log_base={})...".format(log_shift_min, log_base))

        matrix = feature_x_sample.to_numpy()

        if log_shift_min is not None:

            matrix = shift_min(matrix, log_shift_min)

        feature_x_sample = DataFrame(
            data=log(
                matrix,
                log_base=log_base,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **kwarg_)

    if normalize_method is not None:

        print("Axis-{} {} normalizing...".format(normalize_axis, normalize_method))

        feature_x_sample = DataFrame(
            data=normalize(
                feature_x_sample.to_numpy(), normalize_axis, normalize_method
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **kwarg_)

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        feature_x_sample = feature_x_sample.clip(lower=clip_min, upper=clip_max)

        summarize(feature_x_sample, **kwarg_)

    return feature_x_sample
