from ..python import cast_builtin


def _update_with_suffix(an_fe_sa):

    n_fe, n_sa = an_fe_sa.shape

    for ief in range(n_fe):

        for ies in range(n_sa):

            an = an_fe_sa[ief, ies]

            if isinstance(an, str):

                an_fe_sa[ief, ies] = cast_builtin(an.split(sep=": ", maxsplit=1)[1])
