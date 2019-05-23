from pandas import concat


def select_and_group_feature_x_tcga_sample_by_sample_type(
    feature_x_tcga_sample, sample_type
):

    sample_type_selected = feature_x_tcga_sample.loc[
        :, feature_x_tcga_sample.columns.str[13:15] == sample_type
    ]

    print(f"{sample_type_selected.shape}: sample_type_selected")

    duplicated = sample_type_selected.columns.str[:12].duplicated(keep=False)

    sample_type_selected_not_duplicated = sample_type_selected.loc[:, ~duplicated]

    sample_type_selected_not_duplicated.columns = sample_type_selected_not_duplicated.columns.str[
        :12
    ]

    print(
        f"{sample_type_selected_not_duplicated.shape}: sample_type_selected_not_duplicated"
    )

    sample_type_selected_duplicated = sample_type_selected.loc[:, duplicated]

    if sample_type_selected_duplicated.size:

        print(
            f"{sample_type_selected_duplicated.shape}: sample_type_selected_duplicated"
        )

        sample_type_selected_duplicated = sample_type_selected_duplicated.groupby(
            by=sample_type_selected_duplicated.columns.str[:12], axis=1
        ).median()

    return concat(
        (sample_type_selected_not_duplicated, sample_type_selected_duplicated), axis=1
    )
