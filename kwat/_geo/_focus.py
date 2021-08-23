from pandas import Index


def _focus(
    feature_x_sample,
):

    feature_x_sample = feature_x_sample.loc[
        (
            label[:22] == "Sample_characteristics"
            for label in feature_x_sample.index.values
        ),
        :,
    ]

    prefix__ = tuple(_get_prefix(row) for row in feature_x_sample.values)

    if all(len(prefix_) == 1 for prefix_ in prefix__):

        _update_with_suffix(feature_x_sample.values)

        feature_x_sample.index = Index(
            data=(prefix_[0] for prefix_ in prefix__), name=feature_x_sample.index.name
        )

    return feature_x_sample
