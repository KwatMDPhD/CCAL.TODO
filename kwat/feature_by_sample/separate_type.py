from pandas import DataFrame, concat

from ..array import guess_type
from ..python import cast_builtin
from ..series import binarize


def separate_type(nu_fe_sa, dr=True, pr=True):

    co__ = []

    bi_in_sa_ = []

    for _, nu_ in nu_fe_sa.iterrows():

        if dr and nu_.unique().size == 1:

            continue

        try:

            bo = guess_type(nu_.dropna().astype(float).values) == "continuous"

        except ValueError:

            bo = False

        if bo:

            co__.append(nu_.apply(cast_builtin))

        else:

            bi_in_sa = binarize(nu_)

            if pr:

                te = "{}.{{}}".format(bi_in_sa.index.name)

            else:

                te = "{}"

            bi_in_sa.index = [te.format(la) for la in bi_in_sa.index]

            bi_in_sa_.append(bi_in_sa)

    te = "{} ({{}})".format(nu_fe_sa.index.name)

    if 0 < len(co__):

        co_in_sa = DataFrame(co__)

        co_in_sa.index.name = te.format("continuous")

    else:

        co_in_sa = None

    if 0 < len(bi_in_sa_):

        bi_in_sa = concat(bi_in_sa_)

        bi_in_sa.index.name = te.format("binary")

    else:

        bi_in_sa = None

    return co_in_sa, bi_in_sa
