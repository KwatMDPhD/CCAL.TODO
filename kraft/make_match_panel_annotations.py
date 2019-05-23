from pandas import DataFrame


def make_match_panel_annotations(score_moe_p_value_fdr):

    annotations = DataFrame(index=score_moe_p_value_fdr.index)

    if score_moe_p_value_fdr["0.95 MoE"].isna().all():

        annotations["Score"] = score_moe_p_value_fdr["Score"].apply("{:.2f}".format)

    else:

        annotations["Score(\u0394)"] = score_moe_p_value_fdr[
            ["Score", "0.95 MoE"]
        ].apply(lambda score_moe: f"{score_moe[0]:.2f}({score_moe[1]:.2f})", axis=1)

    if not score_moe_p_value_fdr["P-Value"].isna().all():

        function = "{:.2e}".format

        annotations["P-Value"] = score_moe_p_value_fdr["P-Value"].apply(function)

        annotations["FDR"] = score_moe_p_value_fdr["FDR"].apply(function)

    return annotations
