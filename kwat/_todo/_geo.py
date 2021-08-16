from gzip import open

from numpy import array
from pandas import DataFrame, Index
from pandas.api.types import is_numeric_dtype

from .dictionary import summarize
from .internet import download
from .name_biology import name_gene
from .pd import collapse, peek, separate_type
from .support import cast_builtin


def _parse_block(
    block,
):

    dict_table = block.split("_table_begin\n", 1)

    dict = {}

    for line in dict_table[0].splitlines()[:-1]:

        (key, value) = line[1:].split(" = ", 1)

        if key in dict:

            key_original = key

            index = 2

            while key in dict:

                key = "{}_{}".format(key_original, index)

                index += 1

        dict[key] = value

    if len(dict_table) == 2:

        row_ = tuple(line.split("\t") for line in dict_table[1].splitlines()[:-1])

        dict["table"] = DataFrame(
            data=(row[1:] for row in row_[1:]),
            index=(row[0] for row in row_[1:]),
            columns=row_[0][1:],
        )

    return dict


def _name_feature(feature_, platform, platform_table):

    print(platform)

    platform = int(platform[3:])

    if platform in (96, 97, 570):

        label = "Gene Symbol"

        def function(
            name,
        ):

            if name != "":

                return name.split(" /// ", 1)[0]

    elif platform in (13534,):

        label = "UCSC_RefGene_Name"

        def function(
            name,
        ):

            return name.split(";", 1)[0]

    elif platform in (5175, 11532):

        label = "gene_assignment"

        def function(
            name,
        ):

            if isinstance(name, str) and name not in ("", "---"):

                return name.split(" // ", 2)[1]

    elif platform in (2004, 2005, 3718, 3720):

        label = "Associated Gene"

        def function(
            name,
        ):

            return name.split(" // ", 1)[0]

    elif platform in (10558,):

        label = "Symbol"

        function = None

    elif platform in (16686,):

        label = "GB_ACC"

        function = None

    else:

        label = None

        function = None

    for _label in platform_table.columns.values:

        if _label == label:

            print(">> {} <<".format(_label))

        else:

            print(_label)

    if label is None:

        return feature_

    else:

        name_ = platform_table.loc[:, label].values

        if callable(function):

            name_ = array(tuple(function(name) for name in name_))

        feature_to_name = dict(zip(feature_, name_))

        summarize(feature_to_name)

        return array(tuple(feature_to_name.get(feature) for feature in feature_))


def _get_prefix(
    row,
):

    return tuple(
        set(value.split(": ", 1)[0] for value in row if isinstance(value, str))
    )


def _update_with_suffix(
    array_2d,
):

    (axis_0_size, axis_1_size) = array_2d.shape

    for index_0 in range(axis_0_size):

        for index_1 in range(axis_1_size):

            value = array_2d[index_0, index_1]

            if isinstance(value, str):

                array_2d[index_0, index_1] = cast_builtin(value.split(": ", 1)[1])


def _focus(
    feature_x_sample,
):

    feature_x_sample = feature_x_sample.loc[
        (
            label[:22] == "Sample_characteristics"
            for label in feature_x_sample.index.values
        ),
        :,
    ]

    prefix__ = tuple(_get_prefix(row) for row in feature_x_sample.values)

    if all(len(prefix_) == 1 for prefix_ in prefix__):

        _update_with_suffix(feature_x_sample.values)

        feature_x_sample.index = Index(
            data=(prefix_[0] for prefix_ in prefix__), name=feature_x_sample.index.name
        )

    return feature_x_sample


def get_gse(gse_id, directory_path, **kwarg_):

    file_path = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
            gse_id[:-3], gse_id
        ),
        directory_path,
        **kwarg_
    )

    platform_ = {}

    sample_ = {}

    for block in open(file_path, mode="rt", errors="replace").read().split("\n^"):

        (header, block) = block.split("\n", 1)

        block_type = header.split(" = ", 1)[0]

        if block_type in ("PLATFORM", "SAMPLE"):

            print(header)

            dict = _parse_block(block)

            if block_type == "PLATFORM":

                name = dict["Platform_geo_accession"]

                dict["data"] = []

                platform_[name] = dict

            elif block_type == "SAMPLE":

                name = dict["Sample_title"]

                if "table" in dict:

                    value_ = (
                        dict.pop("table")
                        .loc[:, "VALUE"]
                        .replace("", None)
                        .apply(cast_builtin)
                    )

                    if is_numeric_dtype(value_):

                        value_.name = name

                        platform_[dict["Sample_platform_id"]]["data"].append(value_)

                    else:

                        print("VALUE is not numeric:")

                        print(value_.head())

                sample_[name] = dict

    feature_x_sample = DataFrame(data=sample_)

    feature_x_sample.index.name = "Feature"

    _x_sample_ = (feature_x_sample, *separate_type(_focus(feature_x_sample)))

    for _x_sample in _x_sample_:

        if _x_sample is not None:

            peek(_x_sample)

    for (platform, dict) in platform_.items():

        data = dict.pop("data")

        if 0 < len(data):

            _x_sample = DataFrame(data=data).T

            feature_ = _x_sample.index.values

            feature_ = _name_feature(feature_, platform, dict["table"])

            _x_sample.index = Index(data=name_gene(feature_), name="Gene")

            _x_sample = collapse(_x_sample)

            peek(_x_sample)

            _x_sample_ += (_x_sample,)

    return _x_sample_
