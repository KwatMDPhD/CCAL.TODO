from gzip import open

from pandas import DataFrame, Index
from pandas.api.types import is_numeric_dtype

from ..gene import name_gene
from ..internet import download
from ..python import cast_builtin
from .pd import collapse, peek, separate_type


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
