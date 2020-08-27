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

    df_0 = df_[0]

    label_ = df_0.axes[axis]

    for df in df_[1:]:

        label_ = label_.union(df.axes[axis])

    label_ = asarray(sorted(label_))

    return tuple(df.reindex(labels=label_, axis=axis) for df in df_)


# ==============================================================================
# Number DataFrame
# ==============================================================================


def summarize(
    number_df, plot=True, heat_map_max_size=int(1e6), histogram_max_size=int(1e3)
):

    matrix = number_df.to_numpy()

    axis_0_label_ = number_df.index.to_numpy()

    axis_1_label_ = number_df.columns.to_numpy()

    axis_0_name = number_df.index.name

    axis_1_name = number_df.columns.name

    print(matrix.shape)

    matrix_size = matrix.size

    if plot and matrix_size <= heat_map_max_size:

        plot_heat_map(
            matrix, axis_0_label_, axis_1_label_, axis_0_name, axis_1_name,
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
                layout={"xaxis": {"title": {"text": "N NaN"}}},
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
            layout={"xaxis": {"title": {"text": "(Not-NaN) Number"}}},
        )

        plot_histogram(
            (median(matrix, axis=1), median(matrix, axis=0)),
            (axis_0_label_, axis_1_label_),
            (axis_0_name, axis_1_name),
            layout={"xaxis": {"title": {"text": "(Not-NaN) Median"}}},
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
