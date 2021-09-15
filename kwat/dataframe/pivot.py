from numpy import full, nan
from pandas import DataFrame, Index

from ..iterable import map_integer


def pivot(ro_, co_, an_, ron="Row Name", con="Column Name", fu=None):

    ro_it = map_integer(ro_)[0]

    co_it = map_integer(co_)[0]

    an_ro_co = full([len(ro_it), len(co_it)], nan)

    for ro, co, an in zip(ro_, co_, an_):

        ier = ro_it[ro] - 1

        iec = co_it[co] - 1

        if callable(fu):

            an = fu(an_ro_co[ier, iec], an)

        an_ro_co[ier, iec] = an

    return DataFrame(
        data=an_ro_co,
        index=Index(data=ro_it, name=ron),
        columns=Index(data=co_it, name=con),
    )
