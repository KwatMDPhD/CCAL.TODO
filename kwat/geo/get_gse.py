from gzip import open

from pandas import DataFrame, Index
from pandas.api.types import is_numeric_dtype

from ..dataframe import peek
from ..feature_by_sample import collapse, separate_type
from ..gene import rename
from ..internet import download
from ..python import cast_builtin
from ._focus import _focus
from ._name_feature import _name_feature
from ._parse_block import _parse_block


def get_gse(gs, *ar, **ke):

    pa = download(
        "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{0}nnn/{1}/soft/{1}_family.soft.gz".format(
            gs[:-3], gs
        ),
        *ar,
        **ke,
    )

    pl_ = {}

    sa_ = {}

    for bl in open(pa, mode="rt", errors="replace").read().split(sep="\n^"):

        he, bl = bl.split(sep="\n", maxsplit=1)

        ty = he.split(sep=" = ", maxsplit=1)[0]

        if ty in ["PLATFORM", "SAMPLE"]:

            print(he)

            ke_va = _parse_block(bl)

            if ty == "PLATFORM":

                na = ke_va["Platform_geo_accession"]

                ke_va["ro_"] = []

                pl_[na] = ke_va

            elif ty == "SAMPLE":

                na = ke_va["Sample_title"]

                if "table" in ke_va:

                    an_ = (
                        ke_va.pop("table")
                        .loc[:, "VALUE"]
                        .replace("", None)
                        .apply(cast_builtin)
                    )

                    if is_numeric_dtype(an_):

                        an_.name = na

                        pl_[ke_va["Sample_platform_id"]]["ro_"].append(an_)

                    else:

                        print("VALUE is not numeric:")

                        print(an_.head())

                sa_[na] = ke_va

    an_fe_sa = DataFrame(data=sa_)

    an_fe_sa.index.name = "Feature"

    an_fe_sa_ = [an_fe_sa, *separate_type(_focus(an_fe_sa))]

    for an_fe_sa in an_fe_sa_:

        if an_fe_sa is not None:

            peek(an_fe_sa)

    for pl, ke_va in pl_.items():

        ro_ = ke_va.pop("ro_")

        if 0 < len(ro_):

            nu_fe_sa = DataFrame(data=ro_).T

            fe_ = nu_fe_sa.index.values

            fe_ = _name_feature(fe_, pl, ke_va["table"])

            nu_fe_sa.index = Index(data=rename(fe_), name="Gene")

            nu_fe_sa = collapse(nu_fe_sa)

            peek(nu_fe_sa)

            an_fe_sa_.append(nu_fe_sa)

    return an_fe_sa_
