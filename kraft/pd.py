from numpy import asarray, concatenate, isnan, logical_not, median
from numpy.random import choice

from .grid import make_nd
from .plot import plot_heat_map, plot_histogram


def sync(df_, axis):

    df_0 = df_[0]

    label_ = df_0.axes[axis]

    for df in df_[1:]:

        label_ = label_.union(df.axes[axis])

    label_ = asarray(sorted(label_))

    return tuple(df.reindex(labels=label_, axis=axis) for df in df_)


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
