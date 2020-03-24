from os.path import join
from re import sub

import GEOparse
from numpy import nan
from pandas import DataFrame, concat, isna

from .BAD_VALUES import BAD_VALUES
from .drop_dataframe_slice_greedily import drop_dataframe_slice_greedily
from .guess_data_type import guess_data_type
from .make_binary_dataframe_from_categorical_series import (
    make_binary_dataframe_from_categorical_series,
)
from .separate_information_x_sample import separate_information_x_sample


def separate_information_x_sample(information_x_sample, bad_values=BAD_VALUES):

    continuouses = []

    binaries = []

    for information, values in information_x_sample.iterrows():

        values = values.replace(bad_values, nan)

        if 1 < values.dropna().unique().size:

            try:

                is_continuous = guess_data_type(values.astype(float)) == "continuous"

            except ValueError as exception:

                print("({}) {} is not continuous.".format(exception, information))

                is_continuous = False

            if is_continuous:

                continuouses.append(values)

            else:

                binaries.append(make_binary_dataframe_from_categorical_series(values))

    if 0 < len(continuouses):

        continuous_information_x_sample = DataFrame(continuouses)

    else:

        continuous_information_x_sample = None

    if 0 < len(binaries):

        binary_information_x_sample = concat(binaries)

    else:

        binary_information_x_sample = None

    return continuous_information_x_sample, binary_information_x_sample


def clean_and_write_dataframe_to_tsv(dataframe, index_name, tsv_file_path):

    assert not dataframe.index.hasnans

    assert not dataframe.index.has_duplicates

    assert not dataframe.columns.hasnans

    assert not dataframe.columns.has_duplicates

    dataframe = dataframe.fillna(nan)

    dataframe = drop_dataframe_slice_greedily(
        dataframe, None, min_n_not_na_unique_value=1
    )

    dataframe = dataframe.sort_index().sort_index(axis=1)

    dataframe.index.name = index_name

    dataframe.to_csv(tsv_file_path, sep="\t")

    return dataframe


def get_geo(geo_id, directory_path):

    directory_path = "{}/{}".format(directory_path, geo_id)

    print("{}...".format(directory_path))

    gse = GEOparse.get_GEO(geo=geo_id, destdir=directory_path)

    print("Title: {}".format(gse.get_metadata_attribute("title")))

    print("N sample: {}".format(len(gse.get_metadata_attribute("sample_id"))))

    geo_dict = {
        "information_x_sample": None,
        "continuous_information_x_sample": None,
        "binary_information_x_sample": None,
        "id_x_sample": None,
        "id_gene_symbol": None,
        "gene_x_sample": None,
    }

    geo_dict["information_x_sample"] = clean_and_write_dataframe_to_tsv(
        gse.phenotype_data.T,
        "Information",
        join(directory_path, "information_x_sample.tsv"),
    )

    for information_x_sample_name, information_x_sample in zip(
        ("continuous_information_x_sample", "binary_information_x_sample"),
        separate_information_x_sample(
            geo_dict["information_x_sample"].loc[
                geo_dict["information_x_sample"].index.str.startswith("characteristics")
            ]
        ),
    ):

        if information_x_sample is None:

            continue

        information_x_sample.index = (
            sub(r"characteristics_ch\d+.\d+.", "", index)
            for index in information_x_sample.index
        )

        geo_dict[information_x_sample_name] = clean_and_write_dataframe_to_tsv(
            information_x_sample,
            information_x_sample.index.name,
            join(directory_path, "{}.tsv".format(information_x_sample_name)),
        )

    if any(gsm.table.empty for gsm in gse.gsms.values()):

        print(
            "There is at least 1 empty sample. Check for any linked or additional supplementary file in the GEO website."
        )

        return geo_dict

    values = []

    for sample_id, gsm in gse.gsms.items():

        print("Sample: {}".format(sample_id))

        sample_table = gsm.table

        sample_table.columns = sample_table.columns.str.lower().str.replace(" ", "_")

        sample_values = sample_table.set_index("id_ref").squeeze()

        sample_values.name = sample_id

        if isinstance(sample_values, DataFrame):

            sample_values.columns = (
                "{} ({})".format(sample_id, column) for column in sample_values.columns
            )

        values.append(sample_values)

    geo_dict["id_x_sample"] = clean_and_write_dataframe_to_tsv(
        concat(values, axis=1), "ID", join(directory_path, "id_x_sample.tsv")
    )

    id_gene_symbol = None

    for platform_id, gpl in gse.gpls.items():

        print("Platform: {}".format(platform_id))

        platform_table = gpl.table

        platform_table.columns = platform_table.columns.str.lower().str.replace(
            " ", "_"
        )

        platform_table.set_index("id", inplace=True)

        if "gene_symbol" not in platform_table:

            if "gene_assignment" in platform_table.columns:

                gene_symbols = []

                for assignment in platform_table["gene_assignment"]:

                    if not isna(assignment) and "//" in assignment:

                        gene_symbols.append(assignment.split(sep="//")[1].strip())

                    else:

                        gene_symbols.append("NO GENE NAME")

                platform_table["gene_symbol"] = gene_symbols

            elif "oligoset_genesymbol" in platform_table.columns:

                platform_table["gene_symbol"] = platform_table["oligoset_genesymbol"]

            elif "ilmn_gene" in platform_table.columns:

                platform_table["gene_symbol"] = platform_table["ilmn_gene"]

            elif "gene" in platform_table.columns:

                platform_table["gene_symbol"] = platform_table["gene"]

        if "gene_symbol" in platform_table:

            id_gene_symbol = platform_table["gene_symbol"].dropna()

            id_gene_symbol.index = id_gene_symbol.index.astype(str)

            geo_dict["id_gene_symbol"] = id_gene_symbol

            print(
                "N unique gene_symbol: {}".format(
                    (id_gene_symbol.drop_duplicates() != "NO GENE NAME").sum()
                )
            )

            gene_x_sample = geo_dict["id_x_sample"].copy()

            id_gene_symbol = id_gene_symbol.to_dict()

            gene_x_sample.index = gene_x_sample.index.map(
                lambda index: id_gene_symbol.get(str(index), "NO GENE NAME")
            )

            gene_x_sample.drop("NO GENE NAME", errors="ignore", inplace=True)

            gene_x_sample.index.name = "Gene"

            geo_dict["gene_x_sample"] = gene_x_sample

        else:

            print(
                "\tgene_symbol is not a GPL column ({}); IDs may be already gene symbols.".format(
                    platform_table.columns
                )
            )

    return geo_dict
