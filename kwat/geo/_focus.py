from pandas import Index

from ._get_prefix import _get_prefix
from ._update_with_suffix import _update_with_suffix


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
