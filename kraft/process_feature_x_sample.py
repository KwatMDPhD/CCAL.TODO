from numpy import nan
from pandas import DataFrame

from .drop_dataframe_slice import drop_dataframe_slice
from .drop_dataframe_slice_greedily import drop_dataframe_slice_greedily
from .log_array import log_array
from .normalize_dataframe import normalize_dataframe
from .summarize_feature_x_sample import summarize_feature_x_sample


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
    **summarize_feature_x_sample_keyword_arguments,
):

    assert not feature_x_sample.index.has_duplicates

    assert not feature_x_sample.columns.has_duplicates

    summarize_feature_x_sample(
        feature_x_sample, **summarize_feature_x_sample_keyword_arguments
    )

    if 0 < len(features_to_drop):

        features_to_drop = set(features_to_drop) & feature_x_sample.index

        print(
            "Dropping {}: {}...".format(feature_x_sample.index.name, features_to_drop)
        )

        feature_x_sample.drop(features_to_drop, inplace=True)

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if 0 < len(samples_to_drop):

        samples_to_drop = set(samples_to_drop) & feature_x_sample.columns

        print(
            "Dropping {}: {}...".format(feature_x_sample.columns.name, samples_to_drop)
        )

        feature_x_sample.drop(samples_to_drop, axis=1, inplace=True)

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if nanize is not None:

        print("NaNizing <= {}...".format(nanize))

        feature_x_sample[feature_x_sample <= nanize] = nan

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if (
        max_na is not None
        or min_n_not_na_value is not None
        or min_n_not_na_unique_value is not None
    ):

        print("Dropping slice...")

        if drop_axis is None:

            drop_function = drop_dataframe_slice_greedily

        else:

            drop_function = drop_dataframe_slice

        feature_x_sample = drop_function(
            feature_x_sample,
            drop_axis,
            max_na=max_na,
            min_n_not_na_value=min_n_not_na_value,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
        )

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if log_base is not None:

        print(
            "Logging (shift_as_necessary_to_achieve_min_before_logging={}, log_base={})...".format(
                shift_as_necessary_to_achieve_min_before_logging, log_base
            )
        )

        feature_x_sample = DataFrame(
            log_array(
                feature_x_sample.values,
                raise_for_bad=False,
                shift_as_necessary_to_achieve_min_before_logging=shift_as_necessary_to_achieve_min_before_logging,
                log_base=log_base,
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if normalization_method is not None:

        print(
            "Axis-{} {} normalizing...".format(normalization_axis, normalization_method)
        )

        feature_x_sample = normalize_dataframe(
            feature_x_sample, normalization_axis, normalization_method
        )

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        feature_x_sample.clip(lower=clip_min, upper=clip_max, inplace=True)

        summarize_feature_x_sample(
            feature_x_sample, **summarize_feature_x_sample_keyword_arguments
        )

    return feature_x_sample
