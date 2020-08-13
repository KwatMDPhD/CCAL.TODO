from gzip import open

from numpy import asarray
from pandas import DataFrame
from pandas.api.types import is_number

from .dataframe import peak
from .dict_ import summarize
from .feature_x_sample import collapse, separate_type
from .internet import download
from .name_biology import name_genes
from .support import cast_builtin


def get_key_value(line):

    return line[1:-1].split(sep=" = ", maxsplit=1)


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

                table.append(line[:-1].split(sep="\t"))

            break

    table = DataFrame(data=table)

    if not table.empty:

        table.columns = table.iloc[0, :]

        table.drop(labels=table.index.to_numpy()[0], inplace=True)

        table.index = table.iloc[:, 0]

        table.drop(labels=table.columns.to_numpy()[0], axis=1, inplace=True)

        table = table.applymap(cast_builtin)

    dict_["table"] = table

    return dict_


def name_ids(ids, platform, platform_table):

    print(platform)

    platform = int(platform[3:])

    if platform in (96, 97, 570):

        label = "Gene Symbol"

        def function(name):

            if name != "":

                return name.split(sep=" /// ", maxsplit=1)[0]

    elif platform in (13534,):

        label = "UCSC_RefGene_Name"

        def function(name):

            return name.split(sep=";", maxsplit=1)[0]

    elif platform in (5175, 11532):

        label = "gene_assignment"

        def function(name):

            if isinstance(name, str) and name != "---":

                return name.split(sep=" // ", maxsplit=2)[1]

    elif platform in (10558,):

        label = "Symbol"

        function = None

    elif platform in (2004, 3718, 3720):

        label = "Associated Gene"

        def function(name):

            return name.split(sep=" // ", maxsplit=1)[0]

    else:

        label = None

        function = None

    for label_ in platform_table.columns.to_numpy():

        if label_ == label:

            print("> {} <".format(label_))

        else:

            print(label_)

    if label is None:

        return ids

    else:

        names = platform_table.loc[:, label].to_numpy()

        if callable(function):

            names = asarray(tuple(function(name) for name in names))

        id_to_name = dict(zip(ids, names))

        summarize(id_to_name)

        return asarray(tuple(id_to_name.get(id_) for id_ in ids))


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

    continuous_feature_x_sample, binary_feature_x_sample = separate_type(
        feature_x_sample.loc[
            (
                "Sample_characteristics_" in label
                for label in feature_x_sample.index.to_numpy()
            ),
            :,
        ],
        prefix_feature=False,
    )

    _x_samples = (
        feature_x_sample,
        continuous_feature_x_sample,
        binary_feature_x_sample,
    )

    for _x_sample in _x_samples:

        if _x_sample is not None:

            peak(_x_sample)

    for platform, platform_dict in platforms.items():

        sample_values = platform_dict.pop("sample_values")

        if 0 < len(sample_values):

            _x_sample = DataFrame(data=sample_values).T

            _x_sample.index = name_ids(
                _x_sample.index.to_numpy(), platform, platform_dict["table"]
            )

            _x_sample.index = name_genes(_x_sample.index.to_numpy())

            _x_sample.index.name = "Gene"

            _x_sample = collapse(_x_sample)

            _x_sample.columns = (
                sample_id_to_name[id_] for id_ in _x_sample.columns.to_numpy()
            )

            peak(_x_sample)

            _x_samples += (_x_sample,)

    return _x_samples
