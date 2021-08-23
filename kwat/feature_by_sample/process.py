from numpy import apply_along_axis
from pandas import DataFrame

from ..array import log, normalize, shift
from ..dataframe import drop, drop_until


def process(
    nu_fe_sa,
    fe_=(),
    sa_=(),
    na=None,
    axdr=None,
    n_no=None,
    n_un=None,
    lo=None,
    mish=None,
    me=None,
    axno=None,
    mi=None,
    ma=None,
    **ke,
):

    if 0 < len(fe_):

        print("Dropping {}: {}...".format(nu_fe_sa.index.name, fe_))

        nu_fe_sa = nu_fe_sa.drop(fe_, errors="ignore")

        summarize(nu_fe_sa, **ke)

    if 0 < len(sa_):

        print("Dropping {}: {}...".format(nu_fe_sa.columns.name, sa_))

        nu_fe_sa = nu_fe_sa.drop(sa_, 1, errors="ignore")

        summarize(nu_fe_sa, **ke)

    if na is not None:

        print("NaNizing <= {}...".format(na))

        nu_fe_sa[nu_fe_sa <= na] = nan

        summarize(nu_fe_sa, **ke)

    if n_no is not None or n_un is not None:

        print("Dropping (axdr={}, n_no={}, n_un={})...".format(axdr, n_no, n_un))

        if axdr is None:

            dr = drop_until

        else:

            dr = drop

        sh = nu_fe_sa.shape

        nu_fe_sa = dr(
            nu_fe_sa,
            axdr,
            n_no=n_no,
            n_un=n_un,
        )

        if sh != nu_fe_sa.shape:

            summarize(nu_fe_sa, **ke)

    if lo is not None:

        print("Logging (mish={}, lo={})...".format(mish, lo))

        nuar_fe_sa = nu_fe_sa.values

        if mish is not None:

            nuar_fe_sa = shift(nuar_fe_sa, mish)

        nu_fe_sa = DataFrame(
            log(nuar_fe_sa, ba=lo),
            nu_fe_sa.index,
            nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, **ke)

    if me is not None:

        print("Axis-{} {} normalizing...".format(axno, me))

        nu_fe_sa = DataFrame(
            apply_along_axis(normalize, axno, nu_fe_sa.values, me),
            nu_fe_sa.index,
            nu_fe_sa.columns,
        )

        summarize(nu_fe_sa, **ke)

    if mi is not None or ma is not None:

        print("Clipping |{} - {}|...".format(mi, ma))

        nu_fe_sa = nu_fe_sa.clip(mi, ma)

        summarize(nu_fe_sa, **ke)

    return nu_fe_sa
