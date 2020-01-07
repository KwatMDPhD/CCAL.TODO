from os.path import isfile

from pandas import read_csv

from .establish_path import establish_path
from .make_match_panel import make_match_panel
from .normalize_name import normalize_name


def make_match_panels(
    target_x_sample,
    data_dicts,
    directory_path,
    drop_negative_target=False,
    read_score_moe_p_value_fdr=False,
    **make_match_panel_keyword_arguments,
):

    for target_name, target in target_x_sample.iterrows():

        if drop_negative_target:

            target = target[0 < target]

        for data_dict in data_dicts:

            print(
                "Making match panel with target {} and data {}...".format(
                    target_name, data_dict["name"]
                )
            )

            file_path_prefix = "{}/{}/{}".format(
                directory_path,
                normalize_name(target_name),
                normalize_name(data_dict["name"]),
            )

            establish_path(file_path_prefix, "file")

            score_moe_p_value_fdr_file_path = "{}.tsv".format(file_path_prefix)

            if isfile(score_moe_p_value_fdr_file_path) and read_score_moe_p_value_fdr:

                print(
                    "Reading score_moe_p_value_fdr from {}...".format(
                        score_moe_p_value_fdr_file_path
                    )
                )

                score_moe_p_value_fdr = read_csv(
                    score_moe_p_value_fdr_file_path, sep="\t", index_col=0
                )

            else:

                score_moe_p_value_fdr = None

            make_match_panel(
                target,
                data_dict["dataframe"],
                index_x_statistic=score_moe_p_value_fdr,
                data_data_type=data_dict["data_type"],
                layout={"title": {"text": data_dict["name"]}},
                file_path_prefix=file_path_prefix,
                **make_match_panel_keyword_arguments,
            )
