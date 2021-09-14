from gzip import open

from pandas import DataFrame, Index
from pandas.api.types import is_numeric_dtype

from ..dataframe import peek
from ..dictionary import summarize
from ..feature_by_sample import collapse, separate
from ..gene import rename
from ..internet import download
from ..python import cast


def _get_prefix(an_):

    return list(
        set(an.split(sep=": ", maxsplit=1)[0] for an in an_ if isinstance(an, str))
    )


from ..python import cast


def _update_with_suffix(an_fe_sa):

    n_fe, n_sa = an_fe_sa.shape

    for ief in range(n_fe):

        for ies in range(n_sa):

            an = an_fe_sa[ief, ies]

            if isinstance(an, str):

                an_fe_sa[ief, ies] = cast(an.split(sep=": ", maxsplit=1)[1])


def _focus(an_fe_sa):

    an_fe_sa = an_fe_sa.loc[
        [fe.startswith("Sample_characteristics") for fe in an_fe_sa.index.values],
        :,
    ]

    pr__ = [_get_prefix(an_) for an_ in an_fe_sa.values]

    if all(len(pr_) == 1 for pr_ in pr__):

        _update_with_suffix(an_fe_sa.values)

        an_fe_sa.index = Index(data=[pr_[0] for pr_ in pr__], name=an_fe_sa.index.name)

    return an_fe_sa


def _name_feature(fe_, pl, da):

    print(pl)

    pl = int(pl[3:])

    if pl in [96, 97, 570]:

        con = "Gene Symbol"

        def fu(na):

            if na != "":

                return na.split(sep=" /// ", maxsplit=1)[0]

    elif pl in [13534]:

        con = "UCSC_RefGene_Name"

        def fu(na):

            return na.split(sep=";", maxsplit=1)[0]

    elif pl in [5175, 11532]:

        con = "gene_assignment"

        def fu(na):

            if isinstance(na, str) and na not in ["", "---"]:

                return na.split(sep=" // ", maxsplit=2)[1]

    elif pl in [2004, 2005, 3718, 3720]:

        con = "Associated Gene"

        def fu(na):

            return na.split(sep=" // ", maxsplit=1)[0]

    elif pl in [10558]:

        con = "Symbol"

        fu = None

    elif pl in [16686]:

        con = "GB_ACC"

        fu = None

    else:

        con = None

        fu = None

    if con is None:

        return fe_

    for co in da.COLUMNS.values:

        if co == con:

            print(">> {} <<".format(co))

        else:

            print(co)

    na_ = da.loc[:, con].values

    if callable(fu):

        na_ = [fu(na) for na in na_]

    fe_na = dict(zip(fe_, na_))

    summarize(fe_na)

    return [fe_na.get(fe) for fe in fe_]


def _parse_block(bl):

    ke_va = {}

    if "_table_begin" in bl:

        ma, ta = bl.split(sep="_table_begin\n")

        ro_ = [li.split(sep="\t") for li in ta.splitlines()[:-1]]

        ke_va["table"] = DataFrame(
            data=[row[1:] for row in ro_[1:]],
            index=[row[0] for row in ro_[1:]],
            COLUMNS=ro_[0][1:],
        )

    else:

        ma = bl

    # TODO: check -1
    for li in ma.splitlines()[:-1]:

        ke, va = li[1:].split(sep=" = ", maxsplit=1)

        if ke in ke_va:

            keo = ke

            ie = 2

            while ke in ke_va:

                ke = "{}_{}".format(keo, ie)

                ie += 1

        ke_va[ke] = va

    return ke_va


def get(gs, *ar, **ke):

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
                        .replace("null", None)
                        .apply(cast)
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

    an_fe_sa_ = [an_fe_sa, *separate(_focus(an_fe_sa))]

    for an_fe_sa in an_fe_sa_:

        if an_fe_sa is not None:

            peek(an_fe_sa)

    for pl, ke_va in pl_.items():

        ro_ = ke_va.pop("ro_")

        if 0 < len(ro_):

            nu_fe_sa = DataFrame(data=ro_).T

            fe_ = nu_fe_sa.index.values

            fe_ = _name_feature(fe_, pl, ke_va["table"])

            nu_fe_sa.index = Index(data=rename(fe_)[0], name="Gene")

            nu_fe_sa = collapse(nu_fe_sa)

            peek(nu_fe_sa)

            an_fe_sa_.append(nu_fe_sa)

    return an_fe_sa_
