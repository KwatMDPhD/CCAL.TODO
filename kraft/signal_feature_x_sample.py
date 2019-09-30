from pandas import concat

from .normalize_dataframe import normalize_dataframe
from .process_feature_x_sample import process_feature_x_sample
from .select_series_indices import select_series_indices
from .summarize_feature_x_sample import summarize_feature_x_sample


def signal_feature_x_sample(
    feature_x_sample,
    feature_x_sample_alias,
    process_feature_x_sample_keyword_arguments,
    summarize_feature_x_sample_keyword_arguments,
    select_series_index_keyword_arguments,
    signal_type,
    output_directory_path,
):

    feature_x_sample_prepare = process_feature_x_sample(
        feature_x_sample,
        **process_feature_x_sample_keyword_arguments,
        **summarize_feature_x_sample_keyword_arguments,
    )

    tsv_file_path = "{}/{}.tsv".format(output_directory_path, feature_x_sample_alias)

    feature_x_sample_prepare.to_csv(
        tsv_file_path.replace(".tsv", "_prepare.tsv"), sep="\t"
    )

    feature_x_sample_prepare_select_feature = feature_x_sample_prepare.loc[
        select_series_indices(
            feature_x_sample_prepare.std(axis=1),
            ">",
            layout={"yaxis": {"title": {"text": "Standard Deviation"}}},
            html_file_path=tsv_file_path.replace(
                ".tsv", "_prepare_select_feature.html"
            ),
            **select_series_index_keyword_arguments,
        )
    ]

    summarize_feature_x_sample(
        feature_x_sample_prepare_select_feature,
        **summarize_feature_x_sample_keyword_arguments,
    )

    if signal_type == "raw":

        feature_x_sample_prepare_select_feature_signal = (
            feature_x_sample_prepare_select_feature
        )

    elif signal_type == "signed":

        signal_negative = -feature_x_sample_prepare_select_feature.clip(upper=0)

        signal_positive = feature_x_sample_prepare_select_feature.clip(lower=0)

        signal_negative.index = signal_negative.index.map(
            lambda index: "(-) {}".format(index)
        )

        signal_positive.index = signal_positive.index.map(
            lambda index: "(+) {}".format(index)
        )

        feature_x_sample_prepare_select_feature_signal = concat(
            (signal_negative, signal_positive)
        )

    feature_x_sample_prepare_select_feature_signal = normalize_dataframe(
        feature_x_sample_prepare_select_feature_signal, 1, "0-1"
    )

    feature_x_sample_prepare_select_feature_signal.to_csv(
        tsv_file_path.replace(".tsv", "_prepare_select_feature_signal.tsv"), sep="\t"
    )

    summarize_feature_x_sample(
        feature_x_sample_prepare_select_feature_signal,
        **summarize_feature_x_sample_keyword_arguments,
    )

    return feature_x_sample_prepare_select_feature_signal
