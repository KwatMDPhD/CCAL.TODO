from numpy import full
from pandas import DataFrame, Index, notna

from ..iterable import map_integer


def binarize(se):
    an_it = map_integer(se.dropna())[0]

    bi_an_ie = full([len(an_it), se.size], 0)

    for ie, an in enumerate(se.values):
        if notna(an):
            bi_an_ie[an_it[an] - 1, ie] = 1

    return DataFrame(
        data=bi_an_ie, index=Index(data=an_it, name=se.name), columns=se.index
    )
