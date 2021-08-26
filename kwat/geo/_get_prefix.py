def _get_prefix(
    an_,
):

    return list(
        set(an.split(sep=": ", maxsplit=1)[0] for an in an_ if isinstance(an, str))
    )
