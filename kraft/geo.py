from gzip import open

from numpy import asarray
from pandas import DataFrame
from pandas.api.types import is_number

from .dataframe import peak
from .dict_ import summarize
from .feature_x_sample import collapse, separate_type
from .internet import download
from .name_biology import name_genes


def parse_block(io, block_type):

    dict_ = {}

    rows = []

    table_begin = "!{}_table_begin\n".format(block_type)

    table_end = "!{}_table_end\n".format(block_type)

    while True:

        line = io.readline()

        if line == "" or line[0] == "^":

            break

        elif line[0] == "!" and " = " in line:

            key, value = line[1:-1].split(sep=" = ", maxsplit=1)

            dict_[key] = value

        elif line == table_begin:

            while True:

                line = io.readline()

                if line == table_end:

                    break

                rows.append(line[: -len("\n")].split(sep="\t"))

    if 0 < len(rows):

        table = DataFrame(data=rows)

        table.columns = table.iloc[0, :]

        table.drop(labels=[0], inplace=True)

        table.index = table.iloc[:, 0]

        table.drop(labels=table.columns.to_numpy()[0], axis=1, inplace=True)

    else:

        table = None

    dict_["table"] = table

    return dict_


def name_ids(ids, platform, platform_table):

    print(platform)

    platform = int(platform[len("GSE") :])

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

    elif platform in (2004, 2005, 3718, 3720):

        label = "Associated Gene"

        def function(name):

            return name.split(sep=" // ", maxsplit=1)[0]

    else:

        label = None

        function = None

    for label_ in platform_table.columns.to_numpy():

        if label_ == label:

            print(">> {} <<".format(label_))

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

        platform_str = "^PLATFORM"

        platform_str_size = len(platform_str)

        samples = {}

        sample_str = "^SAMPLE"

        sample_str_size = len("^SAMPLE")

        line = None

        while line != "":

            line = io.readline()

            if line[:platform_str_size] == platform_str:

                dict_ = platforms

            elif line[:sample_str_size] == sample_str:

                dict_ = samples

            else:

                continue

            dict_[line.split(sep=" = ", maxsplit=1)[1]] = io.tell()

        for dict_, block_type in ((platforms, "platform"), (samples, "sample")):

            for key in dict_:

                print(key)

                io.seek(dict_[key])

                dict_[key] = parse_block(io, block_type)

            if block_type == "platform":

                dict_["sample_values"] = []

    for sample, sample_dict in samples.items():

        sample_table = sample_dict.pop("table")

        if sample_table is not None:

            values = sample_table.loc[:, "VALUE"]

            if all(is_number(value) for value in values.to_numpy()):

                values.name = sample

                platforms[sample_dict["Sample_platform_id"]]["sample_values"].append(
                    values
                )

    feature_x_sample = DataFrame(data=samples)

    feature_x_sample.index.name = "Feature"

    label = "Sample_title"

    sample_id_to_name = feature_x_sample.loc[label, :].to_dict()

    feature_x_sample.drop(labels=[label], inplace=True)

    feature_x_sample.columns = (
        sample_id_to_name[id_] for id_ in feature_x_sample.columns.to_numpy()
    )

    str_ = "Sample_characteristics_"

    continuous_feature_x_sample, binary_feature_x_sample = separate_type(
        feature_x_sample.loc[
            (label[: len(str_)] == str_ for label in feature_x_sample.index.to_numpy()),
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
