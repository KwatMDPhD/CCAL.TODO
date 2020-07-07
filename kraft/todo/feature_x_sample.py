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
