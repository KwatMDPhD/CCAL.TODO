from numpy import apply_along_axis, nan
from pandas import DataFrame

from ..array import log, normalize
from ..dataframe import drop, drop_while
from .summarize import summarize


def process(
    nu_fe_sa,
    fe_=(),
    sa_=(),
    na=None,
    axd=None,
    n_no=None,
    n_un=None,
    ba=None,
    sh=None,
    me=None,
    axn=None,
    mi=None,
    ma=None,
    **ke_va,
):

    summarize(nu_fe_sa, title="Original", **ke_va)

    te = "Dropping {}: {}"

    if 0 < len(fe_):

        print(te.format(nu_fe_sa.index.name, fe_))

        nu_fe_sa = nu_fe_sa.drop(labels=fe_, errors="ignore")

        summarize(nu_fe_sa, title="Dropped features", **ke_va)

    if 0 < len(sa_):

        print(te.format(nu_fe_sa.columns.name, sa_))

        nu_fe_sa = nu_fe_sa.drop(labels=sa_, axis=1, errors="ignore")

        summarize(nu_fe_sa, title="Dropped samples", **ke_va)

    if na is not None:

        print("NaNizing <= {}".format(na))

        nu_fe_sa[nu_fe_sa <= na] = nan

        summarize(nu_fe_sa, title="NaNized", **ke_va)

    if n_no is not None or n_un is not None:

        print("Dropping (axd={}, n_no={}, n_un={})".format(axd, n_no, n_un))

        if axd is None:

            dr = drop_while

        else:

            dr = drop

        be = nu_fe_sa.shape

        nu_fe_sa = dr(
            nu_fe_sa,
            axd,
            n_no=n_no,
            n_un=n_un,
        )

        if be != nu_fe_sa.shape:

            summarize(nu_fe_sa, title="Dropped", **ke_va)

    if ba is not None:

        print("Logging (ba={}, sh={})...".format(ba, sh))

        nu_fe_sa = DataFrame(
            data=log(nu_fe_sa.values, ba=ba, sh=sh),
            index=nu_fe_sa.index,
            columns=nu_fe_sa.COLUMNS,
        )

        summarize(nu_fe_sa, title="Logged", **ke_va)

    if me is not None:

        print("Axis-{} {} normalizing".format(axn, me))

        nu_fe_sa = DataFrame(
            data=apply_along_axis(normalize, axn, nu_fe_sa.values, me),
            index=nu_fe_sa.index,
            columns=nu_fe_sa.COLUMNS,
        )

        summarize(nu_fe_sa, title="Normalized", **ke_va)

    if mi is not None or ma is not None:

        print("Clipping |{} - {}|".format(mi, ma))

        nu_fe_sa = nu_fe_sa.clip(lower=mi, upper=ma)

        summarize(nu_fe_sa, title="Clipped", **ke_va)

    return nu_fe_sa
