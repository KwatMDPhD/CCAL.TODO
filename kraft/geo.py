from gzip import open
from os.path import isfile

from pandas import DataFrame

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


def get_gse(gse_id, directory_path):

    file_path = "{}/{}_family.soft.gz".format(directory_path, gse_id)

    if not isfile(file_path):

        assert file_path == download(
            "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
                gse_id[:-3], gse_id
            ),
            directory_path,
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

    for sample, dict_ in samples.items():

        sample_table = dict_.pop("table")

        if not sample_table.empty:

            values = sample_table.loc[:, "VALUE"]

            values.name = sample

            platforms[dict_["Sample_platform_id"]]["sample_values"].append(values)

    feature_x_sample = DataFrame(data=samples)

    sample_id_to_name = feature_x_sample.loc["Sample_title", :].to_dict()

    feature_x_sample.columns = feature_x_sample.loc["Sample_title", :]

    feature_x_sample.drop("Sample_title", inplace=True)

    feature_x_sample.index.name = "Feature"

    _x_samples = [feature_x_sample]

    for platform, dict_ in platforms.items():

        _x_sample = DataFrame(data=dict_.pop("sample_values")).T

        if not _x_sample.empty:

            table = dict_["table"]

            def gene_assignment(str_):

                if "//" in str_:

                    return str_.split(sep="//")[1].strip()

            for label, function in (
                ("GB_ACC", None),
                ("gene", None),
                ("gene_assignment", gene_assignment),
                ("gene_symbol", None),
                ("ilmn_gene", None),
                ("oligoset_genesymbol", None),
            ):

                if label in table:

                    genes = table.loc[:, label]

                    if function is not None:

                        genes = genes.apply(function)

                    id_to_gene = genes.to_dict()

                    _x_sample.index = (
                        id_to_gene[id_] for id_ in _x_sample.index.to_numpy()
                    )

                    break

            genes = name_genes(_x_sample.index.to_numpy())

            if genes is not None:

                _x_sample.index = genes

            _x_sample.index.name = platform

            _x_sample.columns = (
                sample_id_to_name[id_] for id_ in _x_sample.columns.to_numpy()
            )

            _x_samples.append(
                _x_sample.loc[~_x_sample.index.isna(), :].groupby(level=0).median()
            )

    for _x_sample in _x_samples:

        print("=" * 80)

        print(_x_sample.shape)

        print("-" * 80)

        print(_x_sample.iloc[:2, :2])

    return tuple(_x_samples)
