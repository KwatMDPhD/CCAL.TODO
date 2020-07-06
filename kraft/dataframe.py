from math import floor

from numpy import apply_along_axis, asarray, full, nan, unique
from numpy.random import choice
from pandas import DataFrame, concat, isna

from .array import guess_type, log, normalize
from .plot import plot_heat_map, plot_histogram
from .series import binarize
from .string import BAD_STR
from .support import cast_builtin, map_objects_to_ints


def check_axes(dataframe):

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

    print("Selected {} label.".format(labels.size))

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

    return_ = False

    while True:

        dataframe = drop_axis_label(
            dataframe,
            axis,
            min_good_value=min_good_value,
            min_good_unique_value=min_good_unique_value,
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


def pivot(
    dataframe, axis_1_label_for_axis_0, axis_1_label_for_axis_1, axis_1_label_for_axis_2
):

    axis_0_labels = unique(dataframe[axis_1_label_for_axis_0].to_numpy())

    axis_0_label_to_i = map_objects_to_ints(axis_0_labels)[0]

    axis_1_labels = unique(dataframe[axis_1_label_for_axis_1].to_numpy())

    axis_1_label_to_i = map_objects_to_ints(axis_1_labels)[0]

    matrix = full((axis_0_labels.size, axis_1_labels.size), nan)

    for axis_0_label, axis_1_label, value in dataframe[
        [axis_1_label_for_axis_0, axis_1_label_for_axis_1, axis_1_label_for_axis_2]
    ].to_numpy():

        matrix[axis_0_label_to_i[axis_0_label], axis_1_label_to_i[axis_1_label]] = value

    return DataFrame(matrix, index=axis_0_labels, columns=axis_1_labels)


def normalize(dataframe, method, normalize_keyword_arguments):

    return DataFrame(
        normalize(dataframe.to_numpy(), method, **normalize_keyword_arguments),
        index=dataframe.index,
        columns=dataframe.columns,
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

    dataframe.to_numpy().flatten()

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


def sample_dataframe(dataframe, axis0_size, axis1_size):

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
