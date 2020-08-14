from gzip import open

from numpy import asarray
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype

from .dataframe import peak
from .dict_ import summarize
from .feature_x_sample import collapse, separate_type
from .internet import download
from .name_biology import name_genes
from .support import cast_builtin


def _make_platform_or_sample_dict(block):

    parts = block.split(sep="_table_begin\n", maxsplit=1)

    dict_ = {}

    for line in parts[0].split(sep="\n"):

        if " = " in line:

            key, value = line[1:].split(sep=" = ", maxsplit=1)

            dict_[key] = value

    if len(parts) == 2:

        rows = tuple(line.split(sep="\t") for line in parts[1].split(sep="\n")[:-1])

        dict_["table"] = DataFrame(
            data=(row[1:] for row in rows[1:]),
            index=(row[0] for row in rows[1:]),
            columns=rows[0][1:],
        )

    return dict_


def _name_features(features, platform, platform_table):

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

    elif platform in (2004, 2005, 3718, 3720):

        label = "Associated Gene"

        def function(name):

            return name.split(sep=" // ", maxsplit=1)[0]

    elif platform in (10558,):

        label = "Symbol"

        function = None

    elif platform in (16686,):

        label = "GB_ACC"

        function = None

    else:

        label = None

        function = None

    for label_ in platform_table.columns.to_numpy():

        if label_ == label:

            print(">> {} <<".format(label_))

        else:

            print(label_)

    if label is None:

        return features

    else:

        names = platform_table.loc[:, label].to_numpy()

        if callable(function):

            names = asarray(tuple(function(name) for name in names))

        id_to_name = dict(zip(features, names))

        summarize(id_to_name)

        return asarray(tuple(id_to_name.get(id_) for id_ in features))


def get_gse(gse_id, directory_path, **download_keyword_arguments):

    file_path = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
            gse_id[:-3], gse_id
        ),
        directory_path,
        **download_keyword_arguments
    )

    platforms = {}

    samples = {}

    for block in open(file_path, mode="rt", errors="replace").read().split(sep="\n^"):

        header, block = block.split(sep="\n", maxsplit=1)

        type_ = header.split(sep=" = ", maxsplit=1)[0]

        if type_ in ("PLATFORM", "SAMPLE"):

            dict_ = _make_platform_or_sample_dict(block)

            if type_ == "PLATFORM":

                name = dict_["Platform_geo_accession"]

                print(name)

                dict_["data"] = []

                platforms[name] = dict_

            elif type_ == "SAMPLE":

                name = dict_["Sample_title"]

                print(name)

                if "table" in dict_:

                    values = (
                        dict_.pop("table")
                        .loc[:, "VALUE"]
                        .replace("", None)
                        .apply(cast_builtin)
                    )

                    if is_numeric_dtype(values):

                        values.name = name

                        platforms[dict_["Sample_platform_id"]]["data"].append(values)

                    else:

                        print("VALUE is not numeric:")

                        print(values.head())

                samples[name] = dict_

    feature_x_sample = DataFrame(data=samples)

    feature_x_sample.index.name = "Feature"

    _x_samples = (
        feature_x_sample,
        *separate_type(
            feature_x_sample.loc[
                (
                    label[:22] == "Sample_characteristics"
                    for label in feature_x_sample.index.to_numpy()
                ),
                :,
            ],
            prefix_feature=False,
        ),
    )

    for _x_sample in _x_samples:

        if _x_sample is not None:

            peak(_x_sample)

    for platform, dict_ in platforms.items():

        data = dict_.pop("data")

        if 0 < len(data):

            _x_sample = DataFrame(data=data).T

            _x_sample.index = _name_features(
                _x_sample.index.to_numpy(), platform, dict_["table"]
            )

            _x_sample.index = name_genes(_x_sample.index.to_numpy())

            _x_sample.index.name = "Gene"

            _x_sample = collapse(_x_sample)

            peak(_x_sample)

            _x_samples += (_x_sample,)

    return _x_samples
