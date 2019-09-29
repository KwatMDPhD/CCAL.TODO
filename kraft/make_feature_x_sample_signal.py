from pandas import concat

from .normalize_dataframe import normalize_dataframe
from .plot_heat_map import plot_heat_map
from .process_feature_x_sample import process_feature_x_sample
from .select_series_indices import select_series_indices
from .summarize_feature_x_sample import summarize_feature_x_sample


def make_feature_x_sample_signal(
    feature_x_sample,
    feature_x_sample_alias,
    process_feature_x_sample_keyword_arguments,
    summarize_feature_x_sample_keyword_arguments,
    select_series_index_keyword_arguments,
    signal_type,
    output_directory_path,
):

    feature_x_sample__process = process_feature_x_sample(
        feature_x_sample,
        **process_feature_x_sample_keyword_arguments,
        **summarize_feature_x_sample_keyword_arguments,
    )

    feature_x_sample__process.to_csv(
        "{}/{}.prepare.tsv".format(output_directory_path, feature_x_sample_alias),
        sep="\t",
    )

    feature_x_sample__process__index = feature_x_sample__process.loc[
        select_series_indices(
            feature_x_sample__process.std(axis=1),
            ">",
            layout={"yaxis": {"title": {"text": "Standard Deviation"}}},
            html_file_path="{}/{}.select_feature.html".format(
                output_directory_path, feature_x_sample_alias
            ),
            **select_series_index_keyword_arguments,
        )
    ]

    summarize_feature_x_sample(
        feature_x_sample__process__index, **summarize_feature_x_sample_keyword_arguments
    )

    if signal_type == "raw":

        feature_x_sample__process__index__signal = feature_x_sample__process__index

    elif signal_type == "signed":

        signal_negative = -feature_x_sample__process__index.clip(upper=0)

        signal_positive = feature_x_sample__process__index.clip(lower=0)

        signal_negative.index = signal_negative.index.map(
            lambda index: "(-) {}".format(index)
        )

        signal_positive.index = signal_positive.index.map(
            lambda index: "(+) {}".format(index)
        )

        feature_x_sample__process__index__signal = concat(
            (signal_negative, signal_positive)
        )

    feature_x_sample__process__index__signal = normalize_dataframe(
        feature_x_sample__process__index__signal, 1, "0-1"
    )

    signal_file_path = "{}/{}.signal.tsv".format(
        output_directory_path, feature_x_sample_alias
    )

    feature_x_sample__process__index__signal.to_csv(signal_file_path, sep="\t")

    summarize_feature_x_sample(
        feature_x_sample__process__index__signal,
        **summarize_feature_x_sample_keyword_arguments,
    )

    plot_heat_map(
        feature_x_sample__process__index__signal,
        html_file_path=signal_file_path.replace(".tsv", ".html"),
    )

    return feature_x_sample__process__index__signal
