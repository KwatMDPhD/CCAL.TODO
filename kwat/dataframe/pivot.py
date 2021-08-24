from numpy import full, nan
from pandas import DataFrame, Index, notna

from ..iterable import map_integer


def pivot(la1_, la2_, an_, na1="Dimension 1", na2="Dimension 2", fu=None):

    la1_it = map_integer(la1_)[0]

    la2_it = map_integer(la2_)[0]

    an_la1_la2 = full([len(la1_it), len(la2_it)], nan)

    for la1, la2, an in zip(la1_, la2_, an_):

        ie1 = la1_it[la1] - 1

        ie2 = la2_it[la2] - 1

        if callable(fu):

            anc = an_la1_la2[ie1, ie2]

            if notna(anc):

                an = fu(anc, an)

        an_la1_la2[ie1, ie2] = an

    return DataFrame(
        data=an_la1_la2,
        index=Index(la1_it, name=na1),
        columns=Index(la2_it, name=na2),
    )
