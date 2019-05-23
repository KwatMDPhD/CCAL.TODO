import GEOparse
from pandas import DataFrame, concat, isna


def download_and_parse_geo(geo_id, directory_path):

    print(f"Downloading {geo_id} into {directory_path} ...")

    gse = GEOparse.get_GEO(geo=geo_id, destdir=directory_path)

    print(f"Title: {gse.get_metadata_attribute('title')}")

    print(f"N sample: {len(gse.get_metadata_attribute('sample_id'))}")

    geo_dict = {
        "id_x_sample": None,
        "id_gene_symbol": None,
        "gene_x_sample": None,
        "information_x_sample": gse.phenotype_data.T,
    }

    print(f"information_x_sample.shape: {geo_dict['information_x_sample'].shape}")

    empty_samples = tuple(
        sample_id for sample_id, gsm in gse.gsms.items() if gsm.table.empty
    )

    if 0 < len(empty_samples):

        print(
            f"Sample(s) ({empty_samples}) are empty (check for any linked or additional supplementary file in the GEO website.)"
        )

        return geo_dict

    values = []

    for sample_id, gsm in gse.gsms.items():

        print(f"{sample_id} ...")

        sample_table = gsm.table

        sample_table.columns = sample_table.columns.str.lower().str.replace(" ", "_")

        sample_values = sample_table.set_index("id_ref").squeeze()

        sample_values.name = sample_id

        if isinstance(sample_values, DataFrame):

            sample_values.columns = (
                f"{sample_id} ({column})" for column in sample_values.columns
            )

        values.append(sample_values)

    geo_dict["id_x_sample"] = concat(values, axis=1).sort_index().sort_index(axis=1)

    print(f"id_x_sample.shape: {geo_dict['id_x_sample'].shape}")

    id_gene_symbol = None

    for platform_id, gpl in gse.gpls.items():

        print(f"{platform_id} ...")

        platform_table = gpl.table

        platform_table.columns = platform_table.columns.str.lower().str.replace(
            " ", "_"
        )

        platform_table.set_index("id", inplace=True)

        if "gene_symbol" not in platform_table.columns:

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

            print(f"id_gene_symbol.shape: {id_gene_symbol.shape}")

            print(f"N valid gene_symbol: {(id_gene_symbol != 'NO GENE NAME').sum()}")

            gene_x_sample = geo_dict["id_x_sample"].copy()

            id_gene_symbol = id_gene_symbol.to_dict()

            gene_x_sample.index = geo_dict["id_x_sample"].index.map(
                lambda index: id_gene_symbol.get(str(index), "NO GENE NAME")
            )

            gene_x_sample.drop("NO GENE NAME", inplace=True, errors="ignore")

            gene_x_sample.index.name = "gene_symbol"

            geo_dict["gene_x_sample"] = gene_x_sample.sort_index().sort_index(axis=1)

            print(f"gene_x_sample.shape: {geo_dict['gene_x_sample'].shape}")

        else:

            print(
                f"\tgene_symbol is not a GPL column ({', '.join(platform_table.columns)}); IDs may be already gene symbols."
            )

    return geo_dict
