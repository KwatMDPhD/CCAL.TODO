def split_and_get(an, se, ie):

    if isinstance(an, str):

        return an.split(sep=se, maxsplit=ie + 1)[ie]

    else:

        return None
