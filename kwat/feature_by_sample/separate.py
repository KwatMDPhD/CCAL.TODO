from pandas import DataFrame, concat

from ..array import guess_type
from ..python import cast
from ..series import binarize


def separate(nu_fe_sa, pr=True):
    se_ = []

    bi_fe_sa_ = []

    for _, se in nu_fe_sa.iterrows():
        if se.unique().size == 1:
            continue

        try:
            co = guess_type(se.dropna().astype(float).values) == "continuous"

        except ValueError:
            co = False

        if co:
            se_.append(se.apply(cast))

        else:
            bi_fe_sa = binarize(se)

            if pr:
                te = "{}.{{}}".format(bi_fe_sa.index.name)

                bi_fe_sa.index = [te.format(fe) for fe in bi_fe_sa.index.values]

            bi_fe_sa_.append(bi_fe_sa)

    te = "{} ({{}})".format(nu_fe_sa.index.name)

    if 0 < len(se_):
        co_fe_sa = DataFrame(data=se_)

        co_fe_sa.index.name = te.format("continuous")

    else:
        co_fe_sa = None

    if 0 < len(bi_fe_sa_):
        bi_fe_sa = concat(bi_fe_sa_, verify_integrity=True)

        bi_fe_sa.index.name = te.format("binary")

    else:
        bi_fe_sa = None

    return co_fe_sa, bi_fe_sa
