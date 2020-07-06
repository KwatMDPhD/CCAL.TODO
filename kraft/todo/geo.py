from gzip import open

from pandas import DataFrame

from .df import group
from .internet import download
from .name_biology import map_genes
from .support import cast_builtin


def parse_block(io, str_):

    dict_ = {}

    table = []

    table_begin_str = "!{}_table_begin\n".format(str_)

    table_end_str = "!{}_table_end\n".format(str_)

    while True:

        line = io.readline()

        if line == "" or line.startswith("^"):

            break

        elif line[0] == "!" and " = " in line:

            key, value = get_key_value(line)

            dict_[key] = value.rstrip("\n")

        elif line == table_begin_str:

            while True:

                line = io.readline()

                if line == table_end_str:

                    break

                table.append(line.rstrip("\n").split(sep="\t"))

            break

    table = DataFrame(data=table)

    if not table.empty:

        table.columns = table.loc[0]

        table = table.drop(0).set_index(table.columns[0]).applymap(cast_builtin)

    dict_["table"] = table

    return dict_


def get_key_value(line):

    return line.lstrip("!").split(sep=" = ", maxsplit=1)


def get_gse(gse_id, directory_path, overwrite=True):

    with open(
        download(
            "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
                gse_id[:-3], gse_id
            ),
            directory_path,
            overwrite=overwrite,
        ),
        mode="rt",
        errors="replace",
    ) as io:

        platforms = {}

        samples = {}

        line = io.readline()

        while line != "":

            line = io.readline()

            if line.startswith("^PLATFORM"):

                platform = get_key_value(line.rstrip("\n"))[1]

                platforms[platform] = io.tell()

            elif line.startswith("^SAMPLE"):

                sample = get_key_value(line.rstrip("\n"))[1]

                samples[sample] = io.tell()

        for dict_, str_ in ((platforms, "platform"), (samples, "sample")):

            for key, offset in dict_.items():

                io.seek(offset)

                dict_[key] = parse_block(io, str_)

    for platform in platforms:

        platforms[platform]["sample_values"] = []

    for sample, sample_dict in samples.items():

        sample_table = sample_dict.pop("table")

        if not sample_table.empty:

            sample_values = sample_table["VALUE"]

            sample_values.name = sample

            platforms[sample_dict["Sample_platform_id"]]["sample_values"].append(
                sample_values
            )

    information_x_sample = DataFrame(samples)

    id_sample_name = information_x_sample.loc["Sample_title"].to_dict()

    information_x_sample.columns = information_x_sample.columns.map(id_sample_name)

    error_axis(information_x_sample, index_name="Information", column_name="Sample")

    _x_samples = [information_x_sample]

    for platform, platform_dict in platforms.items():

        _x_sample = DataFrame(platform_dict.pop("sample_values")).T

        if not _x_sample.empty:

            platform_table = platform_dict["table"]

            def gene_assignment(str_):

                if "//" in str_:

                    return str_.split(sep="//")[1].strip()

            for column, function in (
                ("GB_ACC", None),
                ("gene", None),
                ("gene_assignment", gene_assignment),
                ("gene_symbol", None),
                ("ilmn_gene", None),
                ("oligoset_genesymbol", None),
            ):

                if column in platform_table:

                    series = platform_table[column]

                    if function is not None:

                        series = series.apply(function)

                    _x_sample.index = _x_sample.index.map(series.to_dict())

                    break

            genes = map_genes(_x_sample.index)

            if genes is not None:

                _x_sample.index = genes

            _x_sample.index.name = platform

            _x_sample.columns = _x_sample.columns.map(id_sample_name)

            _x_sample.columns.name = "Sample"

            _x_samples.append(tidy(group(_x_sample.loc[~_x_sample.index.isna()])))

    for _x_sample in _x_samples:

        print("=" * 80)

        print(_x_sample.shape)

        print("-" * 80)

        print(_x_sample.iloc[:2, :2])

    return tuple(_x_samples)
