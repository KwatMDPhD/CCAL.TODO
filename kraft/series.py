from numpy import full
from pandas import DataFrame, Index, notna


def binarize(se):

    anar_ = se.to_numpy()

    an_ie = {}

    ie = 0

    for an in anar_:

        if notna(an) and an not in an_ie:

            an_ie[an] = ie

            ie += 1

    taar = full([len(an_ie), anar_.size], 0)

    for ie2, an in enumerate(anar_):

        if notna(an):

            taar[an_ie[an], ie2] = 1

    return DataFrame(
        taar,
        Index(an_ie, name=se.name),
        se.index,
    )
