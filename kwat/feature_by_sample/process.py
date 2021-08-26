from numpy import apply_along_axis, nan
from pandas import DataFrame

from ..array import log, normalize, shift
from ..dataframe import drop, drop_until
from .summarize import summarize


def process(
    nu_fe_sa,
    fe_=(),
    sa_=(),
    na=None,
    axd=None,
    n_no=None,
    n_un=None,
    lo=None,
    sh=None,
    no=None,
    axn=None,
    mi=None,
    ma=None,
    **ke,
):

    summarize(nu_fe_sa, title="Original", **ke)

    te = "Dropping {}: {}..."

    if 0 < len(fe_):

        print(te.format(nu_fe_sa.index.name, fe_))

        nu_fe_sa = nu_fe_sa.drop(labels=fe_, errors="ignore")

        summarize(nu_fe_sa, title="Dropped features", **ke)

    if 0 < len(sa_):

        print(te.format(nu_fe_sa.columns.name, sa_))

        nu_fe_sa = nu_fe_sa.drop(labels=sa_, axis=1, errors="ignore")

        summarize(nu_fe_sa, title="Dropped samples", **ke)

    if na is not None:

        print("NaNizing <= {}...".format(na))

        nu_fe_sa[nu_fe_sa <= na] = nan

        summarize(nu_fe_sa, title="NaNized", **ke)

    if n_no is not None or n_un is not None:

        print("Dropping (axd={}, n_no={}, n_un={})...".format(axd, n_no, n_un))

        if axd is None:

            dr = drop_until

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

            summarize(nu_fe_sa, title="Dropped", **ke)

    if lo is not None:

        print("Logging (sh={}, lo={})...".format(sh, lo))

        nua_fe_sa = nu_fe_sa.values

        if sh is not None:

            nua_fe_sa = shift(nua_fe_sa, sh)

            summarize(nu_fe_sa, title="Shifted", **ke)

        nu_fe_sa = DataFrame(
            data=log(nua_fe_sa, ba=lo),
            index=nu_fe_sa.index,
            columns=nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, title="Logged", **ke)

    if no is not None:

        print("Axis-{} {} normalizing...".format(axn, no))

        nu_fe_sa = DataFrame(
            data=apply_along_axis(normalize, axn, nu_fe_sa.values, no),
            index=nu_fe_sa.index,
            columns=nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, title="Normalized", **ke)

    if mi is not None or ma is not None:

        print("Clipping |{} - {}|...".format(mi, ma))

        nu_fe_sa = nu_fe_sa.clip(lower=mi, upper=ma)

        summarize(nu_fe_sa, title="Clipped", **ke)

    return nu_fe_sa
