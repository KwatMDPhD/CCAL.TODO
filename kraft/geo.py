from gzip import open

from pandas import DataFrame
from pandas.api.types import is_number

from .dataframe import peak
from .dict_ import summarize
from .internet import download
from .name_biology import name_genes
from .support import cast_builtin


def get_key_value(line):

    return line.rstrip("\n").lstrip("!").split(sep=" = ", maxsplit=1)


def parse_block(io, block_type):

    dict_ = {}

    table = []

    table_begin = "!{}_table_begin\n".format(block_type)

    table_end = "!{}_table_end\n".format(block_type)

    while True:

        line = io.readline()

        if line == "" or line.startswith("^"):

            break

        elif line[0] == "!" and " = " in line:

            key, value = get_key_value(line)

            dict_[key] = value

        elif line == table_begin:

            while True:

                line = io.readline()

                if line == table_end:

                    break

                table.append(line.rstrip("\n").split(sep="\t"))

            break

    table = DataFrame(data=table)

    if not table.empty:

        table.columns = table.iloc[0, :]

        table = table.drop(index=table.index.to_numpy()[0])

        table.index = table.iloc[:, 0]

        table = table.drop(columns=table.columns.to_numpy()[0])

        table = table.applymap(cast_builtin)

    dict_["table"] = table

    return dict_


def get_gse(gse_id, directory_path, **download_keyword_arguments):

    file_path = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
            gse_id[:-3], gse_id
        ),
        directory_path,
        **download_keyword_arguments
    )

    with open(file_path, mode="rt", errors="replace") as io:

        platforms = {}

        samples = {}

        line = io.readline()

        while line != "":

            line = io.readline()

            if line.startswith("^PLATFORM"):

                dict_ = platforms

            elif line.startswith("^SAMPLE"):

                dict_ = samples

            else:

                continue

            dict_[get_key_value(line)[1]] = io.tell()

        for dict_, block_type in ((platforms, "platform"), (samples, "sample")):

            for key in dict_:

                io.seek(dict_[key])

                dict_[key] = parse_block(io, block_type)

    for platform in platforms:

        platforms[platform]["sample_values"] = []

    for sample, sample_dict in samples.items():

        sample_table = sample_dict.pop("table")

        if not sample_table.empty:

            values = sample_table.loc[:, "VALUE"]

            if all(is_number(value) for value in values.to_numpy()):

                values.name = sample

                platforms[sample_dict["Sample_platform_id"]]["sample_values"].append(
                    values
                )

    feature_x_sample = DataFrame(data=samples)

    feature_x_sample.index.name = "Feature"

    sample_id_to_name = feature_x_sample.loc["Sample_title", :].to_dict()

    feature_x_sample.drop(labels=["Sample_title"], inplace=True)

    feature_x_sample.columns = (
        sample_id_to_name[id_] for id_ in feature_x_sample.columns.to_numpy()
    )

    peak(feature_x_sample)

    _x_samples = [feature_x_sample]

    for platform, platform_dict in platforms.items():

        _x_sample = DataFrame(data=platform_dict.pop("sample_values")).T

        if not _x_sample.empty:

            print(platform)

            id_to_gene = {}

            platform_table = platform_dict["table"]

            print("Platform table columns:")

            for label in platform_table.columns.to_numpy():

                print("\t{}".format(label))

            def gene_symbol(str_):

                separator = "///"

                if separator in str_:

                    return str_.split(sep=separator, maxsplit=1)[0].strip()

                return str_

            def gene_assignment(str_):

                separator = "//"

                if separator in str_:

                    return str_.split(sep=separator)[1].strip()

                return str_

            def associated_gene(str_):

                separator = "//"

                if separator in str_:

                    return str_.split(sep=separator)[0].strip()

                return str_

            def ucsc_refgene_name(str_):

                separator = ";"

                if separator in str_:

                    return str_.split(sep=separator, maxsplit=1)[0].strip()

                return str_

            for label, function in (
                ("Gene Symbol", gene_symbol),
                ("Associated Gene", associated_gene),
                ("Symbol", None),
                ("UCSC_RefGene_Name", ucsc_refgene_name),
                ("gene", None),
                ("gene_assignment", gene_assignment),
                ("gene_symbol", None),
                ("ilmn_gene", None),
                ("oligoset_genesymbol", None),
            ):

                if label in platform_table:

                    print(label)

                    genes = platform_table.loc[:, label]

                    if function is not None:

                        genes = genes.apply(function)

                    id_to_gene.update(genes.to_dict())

                    summarize(id_to_gene)

                    break

            _x_sample.index = name_genes(
                tuple(id_to_gene[id_] for id_ in _x_sample.index.to_numpy())
            )

            _x_sample = (
                _x_sample.loc[~_x_sample.index.isna(), :].groupby(level=0).median()
            )

            _x_sample.index.name = "Gene"

            _x_sample.columns = (
                sample_id_to_name[id_] for id_ in _x_sample.columns.to_numpy()
            )

            peak(_x_sample)

            _x_samples.append(_x_sample)

    return tuple(_x_samples)
