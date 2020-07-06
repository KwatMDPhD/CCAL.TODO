from math import floor

from numpy import apply_along_axis, asarray, full, nan, unique
from numpy.random import choice
from pandas import DataFrame, concat, isna

from .array import guess_type, log, normalize
from .plot import plot_heat_map, plot_histogram
from .series import binarize
from .string import BAD_STR
from .support import cast_builtin, map_objects_to_ints


def check_axes(df):

    for axis_labels in (df.index, df.columns):

        labels, counts = unique(axis_labels, return_counts=True)

        is_na = isna(labels)

        assert not is_na.any()

        assert (counts == 1).all()


def sync_axis(dfs, axis, method):

    if method == "union":

        df0 = dfs[0]

        if axis == 0:

            labels = set(df0.index)

        else:

            labels = set(df0.columns)

        for df in dfs[1:]:

            if axis == 0:

                labels = labels.union(set(df.index))

            else:

                labels = labels.union(set(df.columns))

        labels = asarray(sorted(labels))

    elif method == "intersection":

        df0 = dfs[0]

        if axis == 0:

            labels = df0.index.to_list()

        else:

            labels = df0.columns.to_list()

        for df in dfs[1:]:

            if axis == 0:

                labels += df.index.to_list()

            else:

                labels += df.columns.to_list()

        labels, counts = unique(labels, return_counts=True)

        labels = labels[counts == len(dfs)]

    print("Selected {} label.".format(labels.size))

    return tuple(df.reindex(labels, axis=axis) for df in dfs)


def drop_axis_label(df, axis, min_good_value=None, min_good_unique_value=None):

    assert min_good_value is not None or min_good_unique_value is not None

    shape_before = df.shape

    is_kept = full(shape_before[axis], True)

    if axis == 0:

        axis_apply = 1

    elif axis == 1:

        axis_apply = 0

    matrix = df.to_numpy()

    if min_good_value is not None:

        if min_good_value < 1:

            min_good_value = min_good_value * shape_before[axis_apply]

        is_kept &= apply_along_axis(
            lambda vector: min_good_value <= (~isna(vector)).sum(), axis_apply, matrix
        )

    if min_good_unique_value is not None:

        if min_good_unique_value < 1:

            min_good_unique_value = min_good_unique_value * df.shape[axis_apply]

        is_kept &= apply_along_axis(
            lambda vector: min_good_unique_value <= unique(vector[~isna(vector)]).size,
            axis_apply,
            matrix,
        )

    if axis == 0:

        df = df.loc[is_kept, :]

    elif axis == 1:

        df = df.loc[:, is_kept]

    print("{} ==> {}".format(shape_before, df.shape))

    return df


def drop_axes_label(df, axis=None, min_good_value=None, min_good_unique_value=None):

    shape_before = df.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    return_ = False

    while True:

        df = drop_axis_label(
            df,
            axis,
            min_good_value=min_good_value,
            min_good_unique_value=min_good_unique_value,
        )

        shape_after = df.shape

        if return_ and shape_before == shape_after:

            return df

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        return_ = True


def pivot(
    df, axis_1_label_for_axis_0, axis_1_label_for_axis_1, axis_1_label_for_axis_2
):

    axis_0_labels = unique(df[axis_1_label_for_axis_0].to_numpy())

    axis_0_label_to_i = map_objects_to_ints(axis_0_labels)[0]

    axis_1_labels = unique(df[axis_1_label_for_axis_1].to_numpy())

    axis_1_label_to_i = map_objects_to_ints(axis_1_labels)[0]

    matrix = full((axis_0_labels.size, axis_1_labels.size), nan)

    for axis_0_label, axis_1_label, value in df[
        [axis_1_label_for_axis_0, axis_1_label_for_axis_1, axis_1_label_for_axis_2]
    ].to_numpy():

        matrix[axis_0_label_to_i[axis_0_label], axis_1_label_to_i[axis_1_label]] = value

    return DataFrame(matrix, index=axis_0_labels, columns=axis_1_labels)


def normalize(df, method, normalize_keyword_arguments):

    return DataFrame(
        normalize(df.to_numpy(), method, **normalize_keyword_arguments),
        index=df.index,
        columns=df.columns,
    )


def summarize(
    df, plot=True, plot_heat_map_max_size=int(1e6), plot_histogram_max_size=int(1e3)
):

    print("Shape: {}".format(df.shape))

    if plot and df.size <= plot_heat_map_max_size:

        plot_heat_map(df)

    df.to_numpy().flatten()

    df_not_na_values = df.unstack().dropna()

    print("Not-NA min: {:.2e}".format(df_not_na_values.min()))

    print("Not-NA median: {:.2e}".format(df_not_na_values.median()))

    print("Not-NA mean: {:.2e}".format(df_not_na_values.mean()))

    print("Not-NA max: {:.2e}".format(df_not_na_values.max()))

    if plot:

        if plot_histogram_max_size < df_not_na_values.size:

            print("Sampling random {} for histogram...".format(plot_histogram_max_size))

            df_not_na_values = df_not_na_values[
                choice(
                    df_not_na_values.index, size=plot_histogram_max_size, replace=False,
                ).tolist()
                + [df_not_na_values.idxmin(), df_not_na_values.idxmax()]
            ]

        plot_histogram(
            (df_not_na_values,), layout={"xaxis": {"title": {"text": "Not-NA Value"}}},
        )

    df_isna = df.isna()

    n_na = df_isna.values.sum()

    if 0 < n_na:

        axis0_n_na = df_isna.sum(axis=1)

        axis0_n_na.name = df_isna.index.name

        if axis0_n_na.name is None:

            axis0_n_na.name = "Axis 0"

        axis1_n_na = df_isna.sum()

        axis1_n_na.name = df_isna.columns.name

        if axis1_n_na.name is None:

            axis1_n_na.name = "Axis 1"

        if plot:

            plot_histogram(
                (axis0_n_na, axis1_n_na),
                layout={
                    "title": {"text": "Fraction NA: {:.2e}".format(n_na / df.size)},
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


def sample_dataframe(df, axis0_size, axis1_size):

    assert axis0_size is not None or axis1_size is not None

    if axis0_size is not None and axis1_size is not None:

        return df.loc[
            choice(df.index, size=int(floor(df.shape[0] * axis0_size)), replace=False),
            choice(
                df.columns, size=int(floor(df.shape[1] * axis1_size)), replace=False
            ),
        ]

    elif axis0_size is not None:

        return df.loc[
            choice(df.index, size=int(floor(df.shape[0] * axis0_size)), replace=False),
        ]

    else:

        return df[
            choice(df.columns, size=int(floor(df.shape[1] * axis1_size)), replace=False)
        ]


def process_feature_x_sample(
    feature_x_sample,
    features_to_drop=(),
    samples_to_drop=(),
    nanize=None,
    drop_axis=None,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
    shift_as_necessary_to_achieve_min_before_logging=None,
    log_base=None,
    normalization_axis=None,
    normalization_method=None,
    clip_min=None,
    clip_max=None,
    **summarize_keyword_arguments,
):

    assert not feature_x_sample.index.has_duplicates

    assert not feature_x_sample.columns.has_duplicates

    summarize(feature_x_sample, **summarize_keyword_arguments)

    if 0 < len(features_to_drop):

        print(
            "Dropping {}: {}...".format(feature_x_sample.index.name, features_to_drop)
        )

        feature_x_sample.drop(features_to_drop, errors="ignore", iace=True)

        summarize(feature_x_sample, **summarize_keyword_arguments)

    if 0 < len(samples_to_drop):

        print(
            "Dropping {}: {}...".format(feature_x_sample.columns.name, samples_to_drop)
        )

        feature_x_sample.drop(samples_to_drop, axis=1, errors="ignore", iace=True)

        summarize(feature_x_sample, **summarize_keyword_arguments)

    if nanize is not None:

        print("NaNizing <= {}...".format(nanize))

        feature_x_sample[feature_x_sample <= nanize] = nan

        summarize(feature_x_sample, **summarize_keyword_arguments)

    if (
        max_na is not None
        or min_n_not_na_value is not None
        or min_n_not_na_unique_value is not None
    ):

        print("Dropping slice...")

        if drop_axis is None:

            drop_function = drop_axis_label_greedily

        else:

            drop_function = drop_axis_label

        feature_x_sample_shape = feature_x_sample.shape

        feature_x_sample = drop_function(
            feature_x_sample,
            drop_axis,
            min_good=max_na,
            min_n_good_value=min_n_not_na_value,
            min_n_good_unique_value=min_n_not_na_unique_value,
        )

        if feature_x_sample_shape != feature_x_sample.shape:

            summarize(feature_x_sample, **summarize_keyword_arguments)

    if log_base is not None:

        print(
            "Logging (shift_as_necessary_to_achieve_min_before_logging={}, log_base={})...".format(
                shift_as_necessary_to_achieve_min_before_logging, log_base
            )
        )

        feature_x_sample = DataFrame(
            log(
                feature_x_sample.values,
                raise_for_bad=False,
                shift_as_necessary_to_achieve_min_before_logging=shift_as_necessary_to_achieve_min_before_logging,
                log_base=log_base,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **summarize_keyword_arguments)

    if normalization_method is not None:

        print(
            "Axis-{} {} normalizing...".format(normalization_axis, normalization_method)
        )

        feature_x_sample = normalize(
            feature_x_sample, normalization_axis, normalization_method
        )

        summarize(feature_x_sample, **summarize_keyword_arguments)

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        feature_x_sample.clip(lower=clip_min, upper=clip_max, iace=True)

        summarize(feature_x_sample, **summarize_keyword_arguments)

    return feature_x_sample
