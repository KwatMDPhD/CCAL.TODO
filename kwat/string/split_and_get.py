def split_and_get(an, se, ie):

    if isinstance(an, str):

        return an.split(sep=se, maxsplit=max(1, ie))[ie]

    else:

        return None
