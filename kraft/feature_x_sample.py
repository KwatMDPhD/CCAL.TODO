from numpy import nan
from pandas import DataFrame, concat

from .array import guess_type, log, normalize, shift_minimum
from .support import cast_builtin
from .table import binarize, drop_axes_label, drop_axis_label, summarize


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
